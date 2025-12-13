import requests

def check_apis():
    vt_key = "b17c7f821b06c08e25856a31652e9edc0a06c0bbd68f4f2ba4cfb4d0a92605c4"
    urlscan_key = "019b1751-1c9d-72f9-8d64-827b6d69e33b"
    abuseipdb_key = "f7b5a5111073dfd495693f75f018006fa7e4624a0bbcad354757ccaf90e5766cf2c41fdd13fd1432"
    
    # Test VirusTotal
    try:
        response = requests.post("https://www.virustotal.com/vtapi/v2/url/scan",
                               data={'apikey': vt_key, 'url': 'https://httpbin.org/get'}, timeout=5)
        vt = "✅ WORKING" if response.status_code == 200 else "❌ FAILED"
    except:
        vt = "❌ FAILED"
    
    # Test URLScan.io
    try:
        headers = {'API-Key': urlscan_key, 'Content-Type': 'application/json'}
        response = requests.post('https://urlscan.io/api/v1/scan/',
                               headers=headers, json={'url': 'https://httpbin.org/get'}, timeout=5)
        urlscan = "✅ WORKING" if response.status_code == 200 else "❌ FAILED"
    except:
        urlscan = "❌ FAILED"
    
    # Test AbuseIPDB
    try:
        headers = {'Accept': 'application/json', 'Key': abuseipdb_key}
        response = requests.get("https://api.abuseipdb.com/api/v2/check",
                              headers=headers, params={'ipAddress': '8.8.8.8'}, timeout=5)
        abuseipdb = "✅ WORKING" if response.status_code == 200 else "❌ FAILED"
    except:
        abuseipdb = "❌ FAILED"
    
    print(f"VirusTotal:  {vt}")
    print(f"URLScan.io:  {urlscan}")
    print(f"AbuseIPDB:   {abuseipdb}")

check_apis()