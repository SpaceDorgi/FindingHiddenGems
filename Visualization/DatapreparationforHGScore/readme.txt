Script					Required Input Files					Output Files
step1_sqlNewdbAllBus.py			yelp_all_fromcsv.db, users.txt				filtered_yelp_subset.db
step2_calculateHGScore.py		filtered_yelp_subset.db, business_ids.txt		business_metrics.json (or parquet/csv)
step3_MergeHG.py			user_top25_recs.csv, business_metrics.json		user_recs_25_with_gem_score.csv


business_ids.txt = contains all unique business ids in a text file \n separated.
users.txt	= contains all unique user ids in a text file \n separated.
