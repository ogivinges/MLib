import numpy as np
from datetime import datetime


def insert_rows(conn, table, rows, target_fields=None, commit_every=1000):
    if target_fields:
        target_fields = ', '.join(target_fields)
        target_fields = '({})'.format(target_fields)
    else:
        target_fields = ''
    cur = conn.cursor()
    conn.commit()
    i = 0
    for row in rows:
        i += 1
        lst = []
        for cell in row:
            if isinstance(cell, str):
                lst.append("'" + str(cell).replace("'", "''") + "'")
            elif cell is None:
                lst.append('NULL')
            elif isinstance(cell, float) and \
                    np.isnan(cell):  # coerce numpy NaN to NULL
                lst.append('NULL')
            elif isinstance(cell, np.datetime64):
                lst.append("'" + str(cell) + "'")
            elif isinstance(cell, datetime):
                lst.append("to_date('" +
                            cell.strftime('%Y-%m-%d %H:%M:%S') +
                            "','YYYY-MM-DD HH24:MI:SS')")
            else:
                lst.append(str(cell))
        values = tuple(lst)
        sql = 'INSERT /*+ APPEND */ ' \
                'INTO {0} {1} VALUES ({2})'.format(table, target_fields, ','.join(values))
        cur.execute(sql)
        if i % commit_every == 0:
            conn.commit()
    conn.commit()
    cur.close()


def insert_dataframe_to_oracle(df, table_name, conn):
    columns = df.columns.tolist()

    columns_str = ', '.join(columns)
    placeholders = ', '.join([f': {col}' for col in columns])
    insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

    data_to_insert = df.to_dict(orient='records')
    with conn.cursor() as cur:
        cur.executemany(insert_query, data_to_insert)
        conn.commit()
