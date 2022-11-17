SELECT   SUM(L_EXTENDEDPRICE * L_DISCOUNT) AS REVENUE  FROM   LINEITEM  WHERE   L_SHIPDATE >= DATE '1994-01-01'   AND L_SHIPDATE < DATE '1994-01-01' + INTERVAL '1' YEAR   AND L_DISCOUNT BETWEEN 0.06 - 0.01 AND 0.06 + 0.01   AND L_QUANTITY < 24; 
