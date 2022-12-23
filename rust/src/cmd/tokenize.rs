use std::boxed::Box;
use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use clap::Clap;
use rayon::prelude::*;

use crate::cli_utils::{
    acquire_progress_indicator, acquire_tokenizer, get_column_index, ReorderedWriter,
};

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    t.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clap, Debug)]
#[clap(about = "Tokenize tweet text contained in a CSV file.")]
pub struct Opts {
    input: String,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
}

pub fn run(cli_args: &Opts) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.input)?;

    let mut wtr = csv::Writer::from_writer(std::io::stdout());
    write_csv_record!(wtr, ["hash", "tokens"]);

    let bar = acquire_progress_indicator("Tokenizing tweets", cli_args.total);

    let headers = rdr.headers()?;

    let text_column_index = get_column_index(&headers, "text")?;

    let tokenizer = acquire_tokenizer(None)?;
    let reordered_writer = ReorderedWriter::new(&mut wtr);
    let mutex = Mutex::new(reordered_writer);

    rdr.records()
        .enumerate()
        .par_bridge()
        .map(|(i, result)| {
            let record = result.expect("Could not read row!");

            let tokens = tokenizer.tokenize(
                &record
                    .get(text_column_index)
                    .expect("Found a row with fewer columns than expected!"),
                true
            );

            (i, calculate_hash(&tokens), tokens)
        })
        .for_each(|(i, h, tokens)| {
            bar.inc(1);

            let mut locked_wtr = mutex.lock().unwrap();

            locked_wtr.write_vec(i, vec![h.to_string(), tokens.join("|")]);
        });

    bar.finish_at_current_pos();

    Ok(())
}
