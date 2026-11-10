"""General world geography, to tell a wire abroad from a domestic one when the site itself names no places.

Country names (with common short forms), national capitals, the world's largest other cities, continents and
regions, provinces, nationalities and currencies. Never a US state or territory, and never a name that is also a
large US city or county (Birmingham, San Jose, Vancouver, Nassau): those stay domestic. A place that is also a common given name
("Chad", "Jordan") counts only after "in", or with its country after a comma (see retrieve.wire_direction)."""
import re

_COUNTRIES = """Afghanistan|Albania|Algeria|Andorra|Angola|Antigua|Argentina|Armenia|Australia|Austria|Azerbaijan|Bahamas|
Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|Bhutan|Bolivia|Bosnia|Botswana|Brazil|Brunei|Bulgaria|
Burkina Faso|Burundi|Cambodia|Cameroon|Canada|Cape Verde|Central African Republic|Chad|Chile|China|Colombia|Comoros|
Congo|Costa Rica|Croatia|Cuba|Cyprus|Czechia|Czech Republic|Denmark|Djibouti|Dominica|Dominican Republic|Ecuador|
Egypt|El Salvador|England|Equatorial Guinea|Eritrea|Estonia|Eswatini|Ethiopia|Fiji|Finland|France|Gabon|Gambia|
Germany|Ghana|Great Britain|Britain|Greece|Grenada|Guatemala|Guinea|Guyana|Haiti|Holland|Honduras|Hong Kong|Hungary|
Iceland|India|Indonesia|Iran|Iraq|Ireland|Northern Ireland|Israel|Italy|Ivory Coast|Jamaica|Japan|Jordan|Kazakhstan|
Kenya|Kiribati|Korea|South Korea|North Korea|Kosovo|Kuwait|Kyrgyzstan|Laos|Latvia|Lebanon|Lesotho|Liberia|Libya|
Liechtenstein|Lithuania|Luxembourg|Macau|Madagascar|Malawi|Malaysia|Maldives|Mali|Malta|Mauritania|Mauritius|Mexico|
Moldova|Monaco|Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Burma|Namibia|Nauru|Nepal|Netherlands|New Zealand|
Nicaragua|Niger|Nigeria|North Macedonia|Norway|Oman|Pakistan|Palau|Palestine|Panama|Papua New Guinea|Paraguay|Peru|
Philippines|Poland|Portugal|Qatar|Romania|Russia|Rwanda|Saint Lucia|Samoa|San Marino|Saudi Arabia|Scotland|Senegal|
Serbia|Seychelles|Sierra Leone|Singapore|Slovakia|Slovenia|Solomon Islands|Somalia|South Africa|South Sudan|Spain|
Sri Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand|Togo|Tonga|Trinidad|Tobago|
Tunisia|Turkey|Turkmenistan|Tuvalu|Uganda|Ukraine|United Arab Emirates|UAE|United Kingdom|UK|Uruguay|Uzbekistan|
Vanuatu|Vatican|Venezuela|Vietnam|Wales|Yemen|Zambia|Zimbabwe"""

_CAPITALS = """Kabul|Tirana|Algiers|Luanda|Buenos Aires|Yerevan|Canberra|Vienna|Baku|Manama|Dhaka|Bridgetown|Minsk|
Brussels|Belmopan|Thimphu|La Paz|Sarajevo|Gaborone|Brasilia|Sofia|Ouagadougou|Phnom Penh|Yaounde|Ottawa|Santiago|
Beijing|Bogota|Zagreb|Havana|Nicosia|Prague|Copenhagen|Santo Domingo|Quito|Cairo|San Salvador|Tallinn|Addis Ababa|
Suva|Helsinki|Paris|Libreville|Banjul|Berlin|Accra|Athens|Guatemala City|Conakry|Port-au-Prince|Tegucigalpa|Budapest|
Reykjavik|New Delhi|Delhi|Jakarta|Tehran|Baghdad|Dublin|Jerusalem|Tel Aviv|Rome|Kingston|Tokyo|Amman|Astana|Nairobi|
Pristina|Kuwait City|Bishkek|Vientiane|Riga|Beirut|Maseru|Monrovia|Tripoli|Vilnius|Antananarivo|Lilongwe|
Kuala Lumpur|Bamako|Valletta|Nouakchott|Port Louis|Mexico City|Chisinau|Ulaanbaatar|Podgorica|Rabat|Maputo|Windhoek|
Kathmandu|Amsterdam|The Hague|Wellington|Managua|Niamey|Abuja|Pyongyang|Skopje|Oslo|Muscat|Islamabad|Port Moresby|
Asuncion|Lima|Manila|Warsaw|Lisbon|Doha|Bucharest|Moscow|Kigali|Riyadh|Dakar|Belgrade|Freetown|Bratislava|Ljubljana|
Mogadishu|Pretoria|Cape Town|Seoul|Juba|Madrid|Colombo|Khartoum|Paramaribo|Stockholm|Bern|Damascus|Taipei|Dushanbe|
Dodoma|Bangkok|Tunis|Ankara|Ashgabat|Kampala|Kyiv|Kiev|Abu Dhabi|London|Montevideo|Tashkent|Caracas|Hanoi|Sanaa|
Lusaka|Harare|Edinburgh|Cardiff|Belfast"""

_CITIES = """Toronto|Montreal|Calgary|Edmonton|Winnipeg|Quebec City|Guadalajara|Monterrey|Tijuana|Cancun|Puebla|Sao Paulo|
São Paulo|Rio de Janeiro|Medellin|Guayaquil|Lagos|Kano|Johannesburg|Durban|Casablanca|Istanbul|Dubai|Mumbai|Bombay|
Bangalore|Bengaluru|Kolkata|Chennai|Hyderabad|Ahmedabad|Pune|Karachi|Lahore|Chittagong|Shanghai|Guangzhou|Shenzhen|
Chengdu|Wuhan|Tianjin|Chongqing|Osaka|Yokohama|Nagoya|Kyoto|Busan|Ho Chi Minh City|Saigon|Surabaya|Cebu|Davao|Sydney|
Melbourne|Brisbane|Perth|Adelaide|Auckland|Barcelona|Seville|Milan|Naples|Turin|Munich|Frankfurt|Hamburg|Cologne|
Zurich|Geneva|Lyon|Marseille|Manchester|Glasgow|Liverpool|Rotterdam|Antwerp|Krakow|Novosibirsk|Haifa|Jeddah|Porto"""

_REGIONS = """Europe|Asia|Africa|South America|Central America|Latin America|Caribbean|Middle East|Oceania|Scandinavia|
European Union|EU"""

# a province or country after "City, ..." that makes the city foreign
_PROVINCES = """Ontario|Quebec|British Columbia|Alberta|Manitoba|Saskatchewan|Nova Scotia|New Brunswick|Newfoundland|
Jalisco|Bavaria|Catalonia|Andalusia|Lombardy|Tuscany|Queensland|New South Wales|Punjab|Maharashtra|Gujarat"""

# places that are also common given names: only "in <name>" or "<city>, <name>" reads them as places
GIVEN_NAMES = {"chad", "jordan", "israel", "india", "kenya", "sydney", "paris", "adelaide", "santiago", "kingston",
               "florence", "lima", "victoria", "perth", "georgia", "sofia"}

_NATIONALITIES = """Canadian|Mexican|British|English|Scottish|Irish|French|German|Italian|Spanish|Portuguese|Dutch|Belgian|
Swiss|Austrian|Polish|Swedish|Norwegian|Danish|Finnish|Greek|Turkish|Russian|Ukrainian|Chinese|Japanese|Korean|Indian|
Pakistani|Filipino|Philippine|Vietnamese|Thai|Indonesian|Malaysian|Australian|Brazilian|Argentine|Argentinian|
Colombian|Peruvian|Chilean|Venezuelan|Nigerian|Kenyan|Egyptian|Israeli|Jamaican|Haitian|Cuban|Guatemalan|Honduran|
Salvadoran|Nicaraguan|Ecuadorian|Bolivian|Ghanaian|Ethiopian|Moroccan|Saudi|Emirati|Iranian|Iraqi|Lebanese|
Bangladeshi|Nepali|Taiwanese|Singaporean|European"""

_CURRENCIES = r"""euros?|pesos?|rupees?|yen|yuan|renminbi|francs?|kronor|kroner|krona|rand|naira|dirhams?|riyals?|shekels?|
rubles?|roubles?|pounds sterling|british pounds|canadian dollars|australian dollars"""


def _alts(block: str) -> list[str]:
    names = [n.strip() for n in block.replace("\n", "").split("|") if n.strip()]
    return sorted(set(names), key=len, reverse=True)          # longest first: "New South Wales" before "Wales"


FOREIGN = _alts(_COUNTRIES) + _alts(_CAPITALS) + _alts(_CITIES)
FOREIGN_SET = {n.lower() for n in FOREIGN}
REGION_SET = {n.lower() for n in _alts(_REGIONS)}
PROVINCE_SET = {n.lower() for n in _alts(_PROVINCES)} | {n.lower() for n in _alts(_COUNTRIES)}

# "in Toronto", "to the Philippines", "to Europe"; a place must follow the word directly, so "in New Mexico" is not one
PLACE_RE = re.compile(r"\b(in|to|from)\s+(?:the\s+)?(%s)\b"
                      % "|".join(re.escape(n) for n in sorted(FOREIGN + _alts(_REGIONS), key=len, reverse=True)), re.I)
# "Paris, Texas" is domestic; "Toronto, Ontario" and "Lyon, France" are abroad
CITY_COMMA = re.compile(r"\b([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,2}),\s+([A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,2})")
NATIONAL_BANK = re.compile(r"\b(?:%s)\s+(?:\w+\s+)?(?:bank|account|business|company|university|college|school|supplier|"
                           r"vendor|seller)s?\b" % _NATIONALITIES.replace("\n", ""), re.I)
CURRENCY = re.compile(r"\b(?:%s)\b" % _CURRENCIES.replace("\n", ""), re.I)
