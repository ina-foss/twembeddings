#[macro_use]
extern crate lazy_static;

use clap::Clap;

pub mod cli_utils;
pub mod cmd;
pub mod stop_words;
pub mod tokenization;

#[derive(Clap, Debug)]
#[clap(version = "1.0")]
struct Opts {
    #[clap(subcommand)]
    command: SubCommand,
}

#[derive(Clap, Debug)]
enum SubCommand {
    Tokenize(cmd::tokenize::TokenizeOpts),
}

fn main() {
    let cli_args: Opts = Opts::parse();

    let result = match cli_args.command {
        SubCommand::Tokenize(sub_args) => cmd::tokenize::run(&sub_args),
    };

    std::process::exit(match result {
        Ok(_) => 0,
        Err(err) => {
            eprintln!("{}", err);
            1
        }
    })
}
