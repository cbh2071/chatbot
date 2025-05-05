import urllib.parse

# --- 修改 query_val ---
query_val = '(tyrosine kinase) AND organism_name:"Homo sapiens"'
# ----------------------

fields_val = "accession,id,protein_name,organism_name,length"
format_val = "json"
size_val = 10

encoded_query = urllib.parse.quote(query_val)
encoded_fields = urllib.parse.quote(fields_val) # 字段保持不变

new_test_url = f"https://rest.uniprot.org/uniprotkb/search?query={encoded_query}&fields={encoded_fields}&format={format_val}&size={size_val}"

print(new_test_url)