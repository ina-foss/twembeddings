use std::boxed::Box;
use std::collections::HashMap;
use std::error::Error;

use clap::Clap;
use serde::Deserialize;

use crate::cli_utils::{acquire_progress_indicator, acquire_tokenizer};
use crate::vectorization::vectorize;

#[derive(Debug, Deserialize)]
struct VocRecord {
    token: String,
    df: usize,
    idf: f64,
}

#[derive(Clap, Debug)]
#[clap(about = "Find tweet nearest neighbors contained within a given window.")]
pub struct Opts {
    voc_input: String,
    column: String,
    input: String,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
}

pub fn run(cli_args: &Opts) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(&cli_args.voc_input)?;

    // let mut wtr = csv::Writer::from_writer(std::io::stdout());

    let bar = acquire_progress_indicator("Compiling vocabulary", None);

    // 1. Compiling vocabulary
    let mut vocabulary: HashMap<String, (usize, f64)> = HashMap::new();

    for (i, result) in rdr.deserialize().enumerate() {
        bar.inc(1);
        let record: VocRecord = result?;
        vocabulary.insert(record.token, (i, record.idf));
    }

    bar.finish_at_current_pos();

    eprintln!("Found {:?} distinct tokens.", vocabulary.len());

    // 2. Streaming tweets, vectorizing and clustering
    let bar = acquire_progress_indicator("Processing tweets", cli_args.total);

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.input)?;

    let headers = rdr.headers()?;

    let column_index = headers.iter().position(|v| v == cli_args.column);

    if column_index.is_none() {
        return Err(format!(
            "Column \"{}\" does not exist in given CSV file!",
            cli_args.column
        )
        .into());
    }

    let column_index = column_index.unwrap();
    let tokenizer = acquire_tokenizer();

    for result in rdr.records() {
        bar.inc(1);

        let record = result?;

        let text_cell = record
            .get(column_index)
            .expect("Found a row with fewer columns than expected!");

        let tokens = tokenizer.tokenize(&text_cell).collect::<Vec<String>>();
        let vector = vectorize(&vocabulary, &tokens);
    }

    bar.finish_at_current_pos();

    Ok(())
}
