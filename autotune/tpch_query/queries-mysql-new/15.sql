select current_timestamp(6) into @query_start;
set @query_name='15.sql';
CREATE VIEW REVENUE0 (SUPPLIER_NO, TOTAL_REVENUE) AS SELECT L_SUPPKEY, SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) FROM LINEITEM WHERE L_SHIPDATE >= DATE '1997-07-01' AND L_SHIPDATE < DATE '1997-07-01' + INTERVAL '3' MONTH GROUP BY L_SUPPKEY; SELECT S_SUPPKEY, S_NAME, S_ADDRESS, S_PHONE, TOTAL_REVENUE FROM SUPPLIER, REVENUE0 WHERE S_SUPPKEY = SUPPLIER_NO AND TOTAL_REVENUE = ( SELECT MAX(TOTAL_REVENUE) FROM REVENUE0) ORDER BY S_SUPPKEY; DROP VIEW REVENUE0;
set @query_time_ms= timestampdiff(microsecond, @query_start, current_timestamp(6))/1000;
SELECT @query_name, @query_time_ms;