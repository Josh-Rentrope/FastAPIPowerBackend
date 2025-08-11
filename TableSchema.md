==================================================
 Table: conversations
--------------------------------------------------
 COLUMN     | DATA_TYPE
 -----------+-----------
 id         | uuid
 created_at | timestamp with time zone
 updated_at | timestamp with time zone
==================================================

==================================================
 Table: geography_columns
--------------------------------------------------
 COLUMN             | DATA_TYPE
 -------------------+-----------
 f_table_catalog    | name
 f_table_schema     | name
 f_table_name       | name
 f_geography_column | name
 coord_dimension    | integer
 srid               | integer
 type               | text
==================================================

==================================================
 Table: geometry_columns
--------------------------------------------------
 COLUMN            | DATA_TYPE
 ------------------+-----------
 f_table_catalog   | character varying
 f_table_schema    | name
 f_table_name      | name
 f_geometry_column | name
 coord_dimension   | integer
 srid              | integer
 type              | character varying
==================================================

==================================================
 Table: locations
--------------------------------------------------
 COLUMN           | DATA_TYPE
 -----------------+-----------
 id               | integer
 location_text    | text
 geom             | USER-DEFINED
 last_geocoded_at | timestamp with time zone
==================================================

==================================================
 Table: messages
--------------------------------------------------
 COLUMN          | DATA_TYPE
 ----------------+-----------
 id              | uuid
 conversation_id | uuid
 sender_id       | character varying
 content         | text
 sent_at         | timestamp with time zone
==================================================

==================================================
 Table: participants
--------------------------------------------------
 COLUMN          | DATA_TYPE
 ----------------+-----------
 id              | uuid
 conversation_id | uuid
 user_id         | character varying
 joined_at       | timestamp with time zone
==================================================

==================================================
 Table: processed_data
--------------------------------------------------
 COLUMN            | DATA_TYPE
 ------------------+-----------
 nct_number        | text
 study_title       | text
 study_url         | text
 status            | character varying
 conditions        | text
 interventions     | text
 sponsor           | text
 sex               | character varying
 age_group         | character varying
 phase             | character varying
 enrollment_count  | integer
 study_type        | character varying
 start_date        | date
 completion_date   | date
 first_posted_date | date
 last_updated_date | date
 location_geom     | USER-DEFINED
==================================================

==================================================
 Table: raw_data
--------------------------------------------------
 COLUMN                  | DATA_TYPE
 ------------------------+-----------
 id                      | bigint
 nct_number              | text
 study_title             | text
 study_url               | text
 acronym                 | text
 study_status            | text
 brief_summary           | text
 study_results           | text
 conditions              | text
 interventions           | text
 sponsor                 | text
 collaborators           | text
 sex                     | text
 age                     | text
 phases                  | text
 enrollment              | text
 study_type              | text
 other_ids               | text
 start_date              | text
 primary_completion_date | text
 completion_date         | text
 first_posted            | text
 last_update_posted      | text
 locations               | text
 ingested_at             | timestamp with time zone
 source_file_name        | text
==================================================

==================================================
 Table: spatial_ref_sys
--------------------------------------------------
 COLUMN    | DATA_TYPE
 ----------+-----------
 srid      | integer
 auth_name | character varying
 auth_srid | integer
 srtext    | character varying
 proj4text | character varying
==================================================

==================================================
 Table: study_locations
--------------------------------------------------
 COLUMN           | DATA_TYPE
 -----------------+-----------
 study_nct_number | character varying
 location_id      | integer
==================================================

==================================================
 Table: users
--------------------------------------------------
 COLUMN        | DATA_TYPE
 --------------+-----------
 clerk_user_id | character varying
 role          | character varying
 created_at    | timestamp with time zone
==================================================