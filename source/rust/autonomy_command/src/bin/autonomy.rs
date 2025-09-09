use mimalloc::MiMalloc;
use ockam_command::util::exitcode;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    if let Err(e) = ockam_command::entry_point::run() {
        // print initialization errors
        eprintln!("{:?}", e);
        std::process::exit(exitcode::SOFTWARE);
    }
}
