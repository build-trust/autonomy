FROM 1password/connect-api:latest AS api
FROM 1password/connect-sync:latest AS sync

FROM debian:bookworm-slim

# Install supervisor and ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
  supervisor \
  ca-certificates \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /home/opuser/.op/data /var/log/supervisor

# Copy binaries from 1Password images
COPY --from=api /bin/connect-api /usr/local/bin/connect-api
COPY --from=sync /bin/connect-sync /usr/local/bin/connect-sync

# Copy credentials file
COPY 1password-credentials.json /home/opuser/.op/1password-credentials.json
RUN chmod 600 /home/opuser/.op/1password-credentials.json

# Create supervisor configuration
RUN cat > /etc/supervisor/conf.d/op-connect.conf << 'EOF'
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid
user=root

[program:connect-sync]
command=/usr/local/bin/connect-sync
autostart=true
autorestart=true
priority=1
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
environment=OP_SESSION="/home/opuser/.op/1password-credentials.json",XDG_DATA_HOME="/home/opuser/.op"

[program:connect-api]
command=/usr/local/bin/connect-api
autostart=true
autorestart=true
priority=10
startsecs=5
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
environment=OP_SESSION="/home/opuser/.op/1password-credentials.json",XDG_DATA_HOME="/home/opuser/.op",OP_HTTP_PORT="8180"
EOF

# Expose the API port
EXPOSE 8180

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8180/health || exit 1

# Run supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
