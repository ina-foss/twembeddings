use indicatif::{ProgressBar, ProgressStyle};

pub fn acquire_progress_indicator(total: Option<u64>) -> ProgressBar {
    match total {
        Some(total_count) => {
            let bar = ProgressBar::new(total_count);

            bar.set_style(ProgressStyle::default_bar().template(
                "[{elapsed_precise}] < [{eta_precise}] {per_sec} {bar:70} {pos:>7}/{len:7}",
            ));

            bar
        }
        None => {
            let bar = ProgressBar::new_spinner();

            bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner} [{elapsed_precise}] {per_sec} {pos}"),
            );

            bar
        }
    }
}
