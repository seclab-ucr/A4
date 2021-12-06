import re

def read_file_lines(file_to_read):
    with open(file_to_read, 'r') as myfile:
        return myfile.readlines()

file_path = '/mnt/drive/work/adgraph/filterlists_09152018/easylist.txt'

file_lines = read_file_lines(file_path)
count = 0
ex_count = 0
special = ['\\', '/', '*', '.', '-', '_', '&', '=', '^', '?', ';', '|', ',']
keywords_raw = ['ad', 'ads', 'advert', 'popup', 'banner', 'sponsor', 'iframe', 'googlead', 'adsys', 'adser', 'advertise', 
'redirect', 'popunder', 'punder', 'popout', 'click', 'track', 'play', 'pop', 'prebid', 'bid', 'pb.min', 'affiliate', 'ban', 'delivery',
 'promo','tag', 'zoneid', 'siteid', 'pageid', 'sponser', 'size', 'viewid', 'zone_id', 'google_afc' , 'google_afs']

for line in file_lines:
    if not (line.startswith('||') or line.startswith('@@||') or line.startswith('|') or '##' in line or '#@#' in line or '#?#' in line):
        temp = line.split('$')[0].strip()
        matched = False

        for key in keywords_raw:
            key_matches = [m.start() for m in re.finditer(key, temp.lower(), re.I)]
 
            for key_match in key_matches:
                matched = True                
                break
            
            if matched:
                break
        
        if matched:            
            continue

        pattern = '\d{2,4}[xX_-]\d{2,4}'
        if re.compile(pattern).search(temp):
            continue        

        if temp == '':
            continue
        if temp[0] in special: 
            temp = temp[1:]
        if temp[-1:] in special: 
            temp = temp[:-1]
        if temp[0] == '!':
            ex_count += 1 
            continue
        
        print temp
        count += 1

print count, ex_count
