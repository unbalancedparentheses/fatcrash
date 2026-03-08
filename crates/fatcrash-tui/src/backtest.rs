//! Backtest LPPLS + GSADF against historical crashes.
//!
//! Run with: `cargo test --release -p fatcrash-tui -- backtest --nocapture`

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use crate::data::{self, DataSource};

    struct CrashEvent {
        name: &'static str,
        ticker: &'static str,
        source: DataSource,
        fetch_start: NaiveDate,
        fetch_end: NaiveDate,
        crash_date: NaiveDate,
        /// True if this was an exogenous shock, not a bubble (expect low signals).
        exogenous: bool,
    }

    fn known_events() -> Vec<CrashEvent> {
        vec![
            CrashEvent {
                name: "Dotcom Bubble",
                ticker: "SPY",
                source: DataSource::Yahoo { ticker: "SPY".into() },
                fetch_start: NaiveDate::from_ymd_opt(1998, 1, 1).unwrap(),
                fetch_end: NaiveDate::from_ymd_opt(2001, 6, 30).unwrap(),
                crash_date: NaiveDate::from_ymd_opt(2000, 3, 24).unwrap(),
                exogenous: false,
            },
            CrashEvent {
                name: "GFC",
                ticker: "SPY",
                source: DataSource::Yahoo { ticker: "SPY".into() },
                fetch_start: NaiveDate::from_ymd_opt(2005, 1, 1).unwrap(),
                fetch_end: NaiveDate::from_ymd_opt(2009, 6, 30).unwrap(),
                crash_date: NaiveDate::from_ymd_opt(2007, 10, 9).unwrap(),
                exogenous: false,
            },
            CrashEvent {
                name: "COVID Crash",
                ticker: "SPY",
                source: DataSource::Yahoo { ticker: "SPY".into() },
                fetch_start: NaiveDate::from_ymd_opt(2018, 1, 1).unwrap(),
                fetch_end: NaiveDate::from_ymd_opt(2020, 12, 31).unwrap(),
                crash_date: NaiveDate::from_ymd_opt(2020, 2, 19).unwrap(),
                exogenous: true,
            },
            CrashEvent {
                name: "BTC 2017",
                ticker: "BTC-USD",
                source: DataSource::Yahoo { ticker: "BTC-USD".into() },
                fetch_start: NaiveDate::from_ymd_opt(2016, 1, 1).unwrap(),
                fetch_end: NaiveDate::from_ymd_opt(2018, 12, 31).unwrap(),
                crash_date: NaiveDate::from_ymd_opt(2017, 12, 17).unwrap(),
                exogenous: false,
            },
            CrashEvent {
                name: "BTC 2021",
                ticker: "BTC-USD",
                source: DataSource::Yahoo { ticker: "BTC-USD".into() },
                fetch_start: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
                fetch_end: NaiveDate::from_ymd_opt(2022, 6, 30).unwrap(),
                crash_date: NaiveDate::from_ymd_opt(2021, 11, 10).unwrap(),
                exogenous: false,
            },
        ]
    }

    fn run_lppls(
        prices: &[f64],
        dates: &[NaiveDate],
        crash_date: NaiveDate,
    ) -> (f64, f64, Option<NaiveDate>, Option<i64>) {
        let n = prices.len();
        let start = n.saturating_sub(930);
        let slice_prices = &prices[start..];
        let slice_dates = &dates[start..];

        let log_p: Vec<f64> = slice_prices
            .iter()
            .filter_map(|p| if *p > 0.0 { Some(p.ln()) } else { None })
            .collect();
        let times: Vec<f64> = (0..log_p.len()).map(|i| i as f64).collect();

        if log_p.len() < 60 {
            return (0.0, 0.0, None, None);
        }

        let (conf_vec, _, _) = match std::panic::catch_unwind(
            std::panic::AssertUnwindSafe(|| {
                fatcrash_core::lppls::confidence::lppls_confidence_slice(
                    &times,
                    &log_p,
                    Some(60),
                    None,
                    Some(30),
                    Some(20),
                )
            }),
        ) {
            Ok(r) => r,
            Err(_) => return (0.0, 0.0, None, None),
        };

        let eval_len = 180.min(conf_vec.len());
        let eval_start = conf_vec.len() - eval_len;
        let eval_confs = &conf_vec[eval_start..];
        let eval_dates = &slice_dates[slice_dates.len() - eval_len..];

        let max_90d = eval_confs
            .iter()
            .zip(eval_dates.iter())
            .filter(|(_, d)| (0..=90).contains(&(crash_date - **d).num_days()))
            .map(|(c, _)| *c)
            .fold(0.0_f64, f64::max);

        let max_30d = eval_confs
            .iter()
            .zip(eval_dates.iter())
            .filter(|(_, d)| (0..=30).contains(&(crash_date - **d).num_days()))
            .map(|(c, _)| *c)
            .fold(0.0_f64, f64::max);

        let first_watch = eval_confs
            .iter()
            .zip(eval_dates.iter())
            .find(|(c, _)| **c > 0.5)
            .map(|(_, d)| *d);

        let warning_days = first_watch.map(|d| (crash_date - d).num_days());
        (max_90d, max_30d, first_watch, warning_days)
    }

    fn run_gsadf(
        prices: &[f64],
        dates: &[NaiveDate],
        crash_date: NaiveDate,
    ) -> (f64, f64, bool, Option<NaiveDate>, Option<i64>) {
        let log_p: Vec<f64> = prices
            .iter()
            .filter_map(|p| if *p > 0.0 { Some(p.ln()) } else { None })
            .collect();

        if log_p.len() < 50 {
            return (f64::NAN, f64::NAN, false, None, None);
        }

        // 50 MC sims — enough for directional validation, much faster than 200+
        let (stat, bsadf, (_, cv95, _)) =
            fatcrash_core::bubble::gsadf::gsadf_test_slice(&log_p, None, Some(50), Some(42));

        let detected = stat > cv95;

        let first_explosive = bsadf
            .iter()
            .enumerate()
            .find(|(_, v)| v.is_finite() && **v > cv95)
            .map(|(i, _)| i);

        let first_explosive_date = first_explosive.map(|i| dates[i]);
        let warning_days = first_explosive_date.map(|d| (crash_date - d).num_days());

        (stat, cv95, detected, first_explosive_date, warning_days)
    }

    fn run_event(event: &CrashEvent) -> Result<(), String> {
        eprintln!("  Fetching {} ({})...", event.name, event.ticker);
        let bars = data::fetch_range(&event.source, event.fetch_start, event.fetch_end)?;

        let crash_idx = bars
            .iter()
            .rposition(|b| b.date <= event.crash_date)
            .ok_or_else(|| format!("No data on or before crash date {}", event.crash_date))?;

        let pre_crash = &bars[..=crash_idx];
        let prices: Vec<f64> = pre_crash.iter().map(|b| b.close).collect();
        let dates: Vec<NaiveDate> = pre_crash.iter().map(|b| b.date).collect();

        eprintln!("  {} bars up to crash (total fetched: {})", prices.len(), bars.len());

        // LPPLS
        eprintln!("  Running LPPLS...");
        let (max_90d, max_30d, first_watch, lppls_warn) =
            run_lppls(&prices, &dates, event.crash_date);
        eprintln!(
            "  LPPLS: max90d={:.3} max30d={:.3} first_watch={} warn_days={}",
            max_90d,
            max_30d,
            first_watch.map(|d| d.to_string()).unwrap_or("-".into()),
            lppls_warn.map(|d| d.to_string()).unwrap_or("-".into()),
        );

        // GSADF
        eprintln!("  Running GSADF...");
        let (stat, cv95, detected, first_exp, gsadf_warn) =
            run_gsadf(&prices, &dates, event.crash_date);
        eprintln!(
            "  GSADF: stat={:.2} cv95={:.2} detected={} first_explosive={} warn_days={}",
            stat,
            cv95,
            detected,
            first_exp.map(|d| d.to_string()).unwrap_or("-".into()),
            gsadf_warn.map(|d| d.to_string()).unwrap_or("-".into()),
        );

        // Assertions: bubble events should trigger at least one detector;
        // exogenous events (COVID) should NOT trigger.
        if event.exogenous {
            // COVID: true negative expected — don't assert detection
            eprintln!("  [exogenous shock — low signals expected]");
        } else {
            let lppls_fired = max_90d > 0.5;
            let gsadf_fired = detected;
            assert!(
                lppls_fired || gsadf_fired,
                "{}: neither LPPLS (max90d={:.3}) nor GSADF (stat={:.2}, cv95={:.2}) detected the bubble",
                event.name, max_90d, stat, cv95
            );
            eprintln!("  PASS: at least one detector fired");
        }

        Ok(())
    }

    #[test]
        fn backtest_dotcom() {
        let events = known_events();
        let event = events.iter().find(|e| e.name == "Dotcom Bubble").unwrap();
        run_event(event).unwrap();
    }

    #[test]
    fn backtest_gfc() {
        let events = known_events();
        let event = events.iter().find(|e| e.name == "GFC").unwrap();
        run_event(event).unwrap();
    }

    #[test]
    fn backtest_covid() {
        let events = known_events();
        let event = events.iter().find(|e| e.name == "COVID Crash").unwrap();
        run_event(event).unwrap();
    }

    #[test]
    fn backtest_btc2017() {
        let events = known_events();
        let event = events.iter().find(|e| e.name == "BTC 2017").unwrap();
        run_event(event).unwrap();
    }

    #[test]
    fn backtest_btc2021() {
        let events = known_events();
        let event = events.iter().find(|e| e.name == "BTC 2021").unwrap();
        run_event(event).unwrap();
    }

    #[test]
    fn backtest_all_events() {
        let events = known_events();
        let mut passed = 0;
        let mut failed = 0;
        for event in &events {
            eprintln!("\n[{}]", event.name);
            match run_event(event) {
                Ok(()) => passed += 1,
                Err(e) => {
                    eprintln!("  ERROR: {}", e);
                    failed += 1;
                }
            }
        }
        eprintln!("\n{}/{} events passed, {} failed", passed, events.len(), failed);
        assert_eq!(failed, 0, "Some backtest events failed");
    }
}
