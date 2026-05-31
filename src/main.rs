fn main() {
    if let Err(error) = item_waterway_solver::main_cli(std::env::args().skip(1).collect()) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
