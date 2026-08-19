import psycopg2

# ============================================================
# DATABASE CONFIG (તમારી face_search.py માંથી જ કૉપી કરો)
# ============================================================
DB_CONFIG = {
    "dbname": "face_finder",   # અહીં તમારું ડેટાબેઝ નામ લખો
    "user": "postgres",         # અહીં તમારો યુઝર લખો
    "password": "Jayphoto",      # અહીં તમારો પાસવર્ડ લખો
    "host": "127.0.0.1",
    "port": "5432"
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # ૧. 'faces' ટેબલની બધી કોલમ્સ (Columns) ના નામ બતાવો
    cur.execute("""
        SELECT column_name, data_type photos, label, 
        FROM information_schema.columns,  person, faces, 
        WHERE table_name='faces' 
        ORDER BY ordinal_position;
    """)
    columns = cur.fetchall()
    
    print("\n" + "="*50)
    print("📋 'faces' ટેબલની કોલમ્સ (Columns):")
    print("="*50)
    for col_name, col_type in columns:
        print(f"  🔹 {col_name}  ({col_type})")

    # ૨. ટેબલની પહેલી ૩ પંક્તિઓ (Rows) બતાવો (જેથી ડેટા કેવો છે તે ખબર પડે)
    cur.execute("SELECT * FROM faces LIMIT 3;")
    sample_rows = cur.fetchall()
    
    print("\n" + "="*50)
    print("📸 સેમ્પલ ડેટા (પહેલી ૩ પંક્તિઓ):")
    print("="*50)
    for row in sample_rows:
        print(f"  {row}")

    cur.close()
    conn.close()

except Exception as e:
    print(f"❌ ERROR: {e}")