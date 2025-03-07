with cte1 as (
select 
    *, 
    extract('year' from created_at) as year,
    extract('month' from created_at) as month 
from 'users.csv'
), cte2 as (
    select 
        *, 
        row_number() over (partition by year, month order by random()) as r,
        count(*) over(partition by year, month) as c
    from cte1
) select did, handle, bio, created_at from cte2 where r <= 0.05 * c;