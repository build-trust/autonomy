use super::PyWorker;
use std::sync::Arc;

use crate::errors::ockam_error;
use crate::nodes::started_agents::StartedAgents;

use crate::nodes::logs::py_debug;
use ockam::access_control::{AllowAll, IncomingAccessControl, OutgoingAccessControl};
use ockam::{Address, Context, ContextRouter, Result, Routed, Worker, WorkerBuilder, route};
use pyo3::{PyObject, Python};
use serde_json::Value;

/// Spawner creates and manages workers for an agent.
///
/// The spawner pattern enables one logical agent to have multiple worker instances
/// for parallel processing. Each worker is created lazily on first message and is
/// isolated by (scope, conversation) pair.
///
/// Key responsibilities:
/// - Extract scope and conversation from incoming messages
/// - Create workers on-demand, passing scope/conversation context
/// - Route messages to appropriate workers
/// - Cache workers by routing key for reuse
///
/// This enables scope-aware tool creation where tools can be instantiated with
/// proper isolation (e.g., FilesystemTools with different visibility levels).
pub struct Spawner {
    agent_name: String,
    worker_constructor: Arc<PyObject>,
    key_extractor: Arc<PyObject>,
    started_agents: StartedAgents,
    outgoing_ac: Arc<dyn OutgoingAccessControl>,
}

#[ockam::worker]
impl Worker for Spawner {
    type Message = String;
    type Context = Context;

    async fn handle_message(
        &mut self,
        ctx: &mut Self::Context,
        msg: Routed<Self::Message>,
    ) -> Result<()> {
        let local_msg = msg.local_message().clone();
        let message = msg.into_body()?;

        let key_extractor = self.key_extractor.clone();
        let message_clone = message.clone();

        // Extract routing key for worker selection (used for caching)
        // The key extractor typically returns "{scope}/{conversation}"
        let key: Option<String> = ctx
            .runtime()
            .spawn_blocking(move || {
                Python::with_gil(|py| key_extractor.call1(py, (message_clone,))?.extract(py))
                    .map_err(ockam_error)
            })
            .await
            .unwrap()?;

        let key = match key {
            Some(k) => format!("{}/{}", self.agent_name, k),
            None => self.agent_name.clone(),
        };

        // Extract scope and conversation for worker creation
        // These are passed to the worker constructor to enable scope-aware tools
        // (e.g., FilesystemTools can create instances with proper visibility isolation)
        let (scope, conversation) = extract_scope_and_conversation(&message);

        let worker_address = match self.started_agents.get_worker(&self.agent_name, &key).await {
            Some(address) => address,
            None => {
                let address = self.create_worker(ctx, scope, conversation).await?;
                self.started_agents
                    .add_worker(&self.agent_name, key, &address)
                    .await;
                address
            }
        };

        ctx.forward(local_msg.set_onward_route(route![worker_address]))
            .await
    }
}

impl Spawner {
    pub fn start(
        ctx: &ContextRouter,
        agent_name: &str,
        worker_constructor: PyObject,
        key_extractor: PyObject,
        started_agents: StartedAgents,
        incoming_ac: Arc<dyn IncomingAccessControl>,
        outgoing_ac: Arc<dyn OutgoingAccessControl>,
    ) -> Result<()> {
        let agent_name = agent_name.to_string();
        let address: Address = agent_name.clone().into();
        let worker = Self {
            agent_name,
            worker_constructor: Arc::new(worker_constructor),
            key_extractor: Arc::new(key_extractor),
            started_agents,
            outgoing_ac,
        };

        WorkerBuilder::new(worker)
            .with_address(address)
            .with_incoming_access_control_arc(incoming_ac)
            // Note: The spawner only sends messages to workers it spawned itself,
            // so AllowAll is safe for outgoing access control.
            .with_outgoing_access_control(AllowAll)
            .start_using_router_context(ctx)?;

        Ok(())
    }

    /// Create a new worker with scope and conversation context.
    ///
    /// This method creates a worker by calling the Python worker_constructor
    /// with scope and conversation parameters. This enables the agent to:
    /// - Create scope-specific tools (e.g., FilesystemTools with visibility levels)
    /// - Isolate resources per (scope, conversation) pair
    /// - Maintain separate state for each conversation
    ///
    /// The worker constructor receives:
    /// - scope: User/tenant identifier (e.g., "user-alice")
    /// - conversation: Session identifier (e.g., "chat-1")
    async fn create_worker(
        &self,
        ctx: &mut Context,
        scope: Option<String>,
        conversation: Option<String>,
    ) -> ockam::Result<Address> {
        let address = Address::random_tagged(&self.agent_name);
        let agent_name = self.agent_name.clone();

        let worker_constructor = self.worker_constructor.clone();
        let worker = ctx
            .runtime()
            .spawn_blocking(move || {
                Python::with_gil(|py| {
                    py_debug(
                        py,
                        format!("create a new worker for agent '{}' with scope={:?}, conversation={:?}",
                                agent_name, scope, conversation),
                    )?;

                    // Call worker_constructor with scope and conversation parameters.
                    // This enables scope-aware tool creation in Python:
                    // - Factory tools can create isolated instances per scope/conversation
                    // - FilesystemTools uses these to determine visibility paths
                    // - Each worker gets properly isolated tool instances
                    worker_constructor.call1(py, (scope, conversation))
                })
                .map_err(ockam_error)
            })
            .await
            .unwrap()?;

        WorkerBuilder::new(PyWorker::new(worker))
            .with_address(address.clone())
            .with_outgoing_access_control_arc(self.outgoing_ac.clone())
            .start(ctx)
            .map_err(ockam_error)?;

        Ok(address)
    }
}

/// Extract scope and conversation from a JSON message string.
///
/// Parses the message as JSON and extracts the "scope" and "conversation" fields
/// which are used to create isolated worker instances with scope-aware tools.
///
/// These values enable:
/// - FilesystemTools to determine visibility paths (all/agent/scope/conversation)
/// - Proper multi-tenant isolation
/// - Session-specific resource allocation
///
/// Returns (None, None) if the message is not valid JSON or fields are missing.
/// Tools can default to "default" scope/conversation when None is provided.
fn extract_scope_and_conversation(message: &str) -> (Option<String>, Option<String>) {
    match serde_json::from_str::<Value>(message) {
        Ok(json) => {
            let scope = json.get("scope")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let conversation = json.get("conversation")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            (scope, conversation)
        }
        Err(_) => (None, None),
    }
}
