# Benchmark instances

Distance matrices of the classic traveling tournament problem (`L=1`, `U=3` in the usual notation,
i.e. at most three consecutive home games or away games), one team per row and column.

| Files | Teams | Origin |
|---|---|---|
| `nl*.txt` | 4-16 | National League of Major League Baseball (Easton, Nemhauser, Trick) |
| `circ*.txt` | 4-20 | circular distances |
| `nfl*.txt` | 16-32 | National Football League |

The matrices are taken from Michael Trick's challenge instances
(<https://mat.tepper.cmu.edu/TTP/>). `reference.json` holds the best known lower and upper bounds
of the RobinX travel repository (<https://www.sportscheduling.ugent.be/RobinX/travelRepo.php>,
retrieved in October 2026). These are results of specialised algorithms; the best known upper
bounds of the larger instances are not proven optimal.

Evaluate the solver with `python benchmarks/bench_quality.py`.
