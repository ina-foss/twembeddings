#[macro_use]
extern crate lazy_static;

use clap::Clap;

#[macro_use]
pub mod cli_utils;
pub mod clustering;
pub mod cmd;
pub mod cosine;
pub mod date_utils;
pub mod stop_words;
pub mod tokenization;
pub mod vectorization;

#[derive(Clap, Debug)]
#[clap(version = "1.0")]
struct Opts {
    #[clap(subcommand)]
    command: SubCommand,
}

#[derive(Clap, Debug)]
enum SubCommand {
    Eval(cmd::evaluation::Opts),
    Selfeval(cmd::selfevaluation::Opts),
    Nn(cmd::neighbors::Opts),
    Tok(cmd::tokenize::Opts),
    Vocab(cmd::vocabulary::Opts),
    Window(cmd::window::Opts),
}

fn main() {
    let cli_args: Opts = Opts::parse();

    let result = match cli_args.command {
        SubCommand::Eval(sub_args) => cmd::evaluation::run(&sub_args),
        SubCommand::Selfeval(sub_args) => cmd::selfevaluation::run(&sub_args),
        SubCommand::Nn(sub_args) => cmd::neighbors::run(&sub_args),
        SubCommand::Tok(sub_args) => cmd::tokenize::run(&sub_args),
        SubCommand::Vocab(sub_args) => cmd::vocabulary::run(&sub_args),
        SubCommand::Window(sub_args) => cmd::window::run(&sub_args),
    };

    std::process::exit(match result {
        Ok(_) => 0,
        Err(err) => {
            eprintln!("{}", err);
            1
        }
    })
}
