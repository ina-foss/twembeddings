#[macro_use]
extern crate lazy_static;

use std::boxed::Box;
use std::error::Error;

use clap::Clap;

pub mod cli_utils;
pub mod tokenization;

use cli_utils::acquire_progress_indicator;
use tokenization::Tokenizer;

#[derive(Clap, Debug)]
#[clap(version = "1.0")]
struct Opts {
    #[clap(subcommand)]
    command: SubCommand,
}

#[derive(Clap, Debug)]
enum SubCommand {
    Tokenize(TokenizeOpts),
}

#[derive(Clap, Debug)]
#[clap(about = "Tokenize tweet text contained in a CSV file.")]
struct TokenizeOpts {
    column: String,
    input: String,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
}

fn tokenize(cli_args: &TokenizeOpts) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.input)?;

    let mut wtr = csv::Writer::from_writer(std::io::stdout());
    wtr.write_record(&csv::StringRecord::from(vec!["tokens"]))?;

    let bar = acquire_progress_indicator(cli_args.total);

    let headers = rdr.headers()?;

    let column_index = headers.iter().position(|v| v == cli_args.column);

    if column_index.is_none() {
        Err(format!(
            "Column \"{}\" does not exist in given CSV file!",
            cli_args.column
        ))?
    }

    let column_index = column_index.unwrap();

    let tokenizer = Tokenizer::new();

    for result in rdr.records() {
        bar.inc(1);

        let record = result?;
        let text = record.get(column_index).unwrap();

        let tokens: Vec<String> = tokenizer.tokenize(&text).collect();

        wtr.write_record(&csv::StringRecord::from(vec![tokens.join("|")]))?;
    }

    bar.finish_at_current_pos();

    Ok(())
}

fn main() {
    let cli_args: Opts = Opts::parse();

    let result = match cli_args.command {
        SubCommand::Tokenize(sub_args) => tokenize(&sub_args),
    };

    std::process::exit(match result {
        Ok(_) => 0,
        Err(err) => {
            eprintln!("{}", err);
            1
        }
    })
}
