use std::boxed::Box;
use std::error::Error;

use clap::Clap;

use crate::cli_utils::{acquire_progress_indicator, get_column_index};

#[derive(Clap, Debug)]
#[clap(about = "Infer the size of the window for the clustering algorithm.")]
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

    let bar = acquire_progress_indicator("Processing tweets", cli_args.total);

    let headers = rdr.headers()?;

    let date_column_index = get_column_index(&headers, "created_at")?;

    for result in rdr.records() {
        bar.inc(1);

        let record = result?;

        let date = &record[date_column_index];

        println!("{:?}", date);
    }

    bar.finish_at_current_pos();

    Ok(())
}
