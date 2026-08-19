import pickle
from google_auth_oauthlib.flow import InstalledAppFlow

# ફક્ત Drive ફાઈલો મેનેજ કરવાની પરવાનગી
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def main():
    """credentials.json વાંચીને token.pickle ફાઈલ બનાવો"""
    
    # 'credentials.json' તમે Google Cloud પરથી ડાઉનલોડ કરેલી છે
    flow = InstalledAppFlow.from_client_secrets_file(
        'credentials.json', SCOPES)
    
    # બ્રાઉઝર ખોલીને લોગિન કરાવો
    creds = flow.run_local_server(port=0)
    
    # token.pickle ફાઈલમાં સેવ કરો
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)
    
    print("✅ token.pickle સફળતાપૂર્વક બની ગયું છે!")

if __name__ == '__main__':
    main()