with custom_date_range as (
    select
     unnest(generate_series(timestamp '2022-01-01', timestamp '2025-03-01', interval '1 day'))::date as post_date
),
user_posts_in_custom_date_range as (
    select
        a.author_did,
        b.post_date::date as post_date,
        ap.cid,
    from (select distinct author_did from all_posts) a
    cross join custom_date_range b
    left join all_posts ap on ap.created_at::date = b.post_date and ap.author_did = a.author_did
), daily_counts as (
    select
        author_did,
        post_date,
        coalesce(count(cid), 0) as post_count
    from user_posts_in_custom_date_range
    group by 1,2
    order by 1,2
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