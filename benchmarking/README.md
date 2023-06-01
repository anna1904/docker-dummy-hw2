## RESULTS

| Format         | Load Time       | Save Time       |
|----------------|-----------------|-----------------|
| csv | 0.000970 | 0.003789 |
| json | 0.007822        | 0.000779 |
| pickle | 0.000407        | 0.000455|
| parquet | 0.111923        | 0.128151|
| feather | 0.002378        | 0.012276 |

Conclusion: In row based forlant it is faster to read, but slower to do operations, opposite for column based.

Csv, json, pickle are row based, parquet, feather are column based