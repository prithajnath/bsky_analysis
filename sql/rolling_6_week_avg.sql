with daily_counts as (
    select 
        author_did,
        created_at::date as post_date,
        count(*) as post_count
    from all_posts
    group by 1,2 
),
rolling_avg as (
    select 
        author_did,
        post_date,
        post_count,
        avg(post_count) over (
            partition by author_did
            order by post_date
            rows between 41 preceding and current row
        ) as rolling_6week_avg
    from daily_counts
)
select 
    author_did,
    post_date,
    post_count,
    rolling_6week_avg
from rolling_avg
order by author_did, post_date;