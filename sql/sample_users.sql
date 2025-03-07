with monthly_weights as (
    select '2022-11' as year_month, 0.000196 as value
    union all
    select '2022-12', 0.000123
    union all
    select '2023-1', 0.000733
    union all
    select '2023-10', 0.73771
    union all
    select '2023-11', 0.28432
    union all
    select '2023-12', 0.21694
    union all
    select '2023-2', 0.002495
    union all
    select '2023-3', 0.009879
    union all
    select '2023-4', 0.64784
    union all
    select '2023-5', 0.58271
    union all
    select '2023-6', 0.58510
    union all
    select '2023-7', 1.35349
    union all
    select '2023-8', 0.86979
    union all
    select '2023-9', 0.81314
    union all
    select '2024-1', 0.11714
    union all
    select '2024-10', 0.49473
    union all
    select '2024-11', 1.59647
    union all
    select '2024-12', 0.09356
    union all
    select '2024-2', 0.55653
    union all
    select '2024-3', 0.03677
    union all
    select '2024-4', 0.10437
    union all
    select '2024-5', 0.00227
    union all
    select '2024-6', 0.005135
    union all
    select '2024-7', 0.002060
    union all
    select '2024-8', 0.39101
    union all
    select '2024-9', 0.21019
    union all
    select '2025-1', 0.06354
    union all
    select '2025-2', 0.001617
), cte1 as (
select 
    *, 
    extract('year' from created_at) as year,
    extract('month' from created_at) as month 
from 'unduplicated_users.csv'
), cte2 as (
    select 
        *, 
        row_number() over (partition by year, month order by random()) as r,
        count(*) over(partition by year, month) as c,
        mw.value as weight
    from cte1
    join monthly_weights as mw on mw.year_month = concat(cte1.year::text,'-' ,cte1.month::text)
) select did, handle, bio, created_at from cte2 where r <= weight * c and created_at is not null;