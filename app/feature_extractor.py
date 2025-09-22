import re
import socket
from urllib.parse import urlparse
import numpy as np

def extract_features(url):
    """
    Extracts a dictionary of features from a raw URL.
    """
    features = {}

    parsed = urlparse(url)

    # 1. URL Length
    features["urlLen"] = len(url)

    # 2. Domain Length
    domain = parsed.netloc
    features["domainlength"] = len(domain)

    # 3. Path Length
    features["pathLength"] = len(parsed.path)

    # 4. Number of Dots
    features["NumberofDotsinURL"] = url.count(".")

    # 5. Is IP Address
    try:
        socket.inet_aton(domain)
        features["ISIpAddressInDomainName"] = 1
    except socket.error:
        features["ISIpAddressInDomainName"] = 0

    # 6. Use of HTTPS
    features["Use_of_HTTPS"] = 1 if parsed.scheme == "https" else 0

    # 7. Presence of '@'
    features["Have_At_Symbol"] = 1 if "@" in url else 0

    # 8. Count of Hyphens
    features["Count_of_Hyphen"] = url.count("-")

    # 9. Count of Digits
    features["URL_DigitCount"] = sum(c.isdigit() for c in url)

    # 10. Count of Letters
    features["URL_Letter_Count"] = sum(c.isalpha() for c in url)

    # 11. Entropy (measuring randomness)
    prob = [float(url.count(c)) / len(url) for c in set(url)]
    entropy = -sum([p * np.log2(p) for p in prob])
    features["Entropy_URL"] = entropy

    # 12. Presence of shortening service
    shortening_services = ["bit.ly", "goo.gl", "tinyurl.com", "ow.ly"]
    features["Shortening_Service"] = int(any(service in url for service in shortening_services))

    # 13. Number of subdirectories
    features["subDirLen"] = len([p for p in parsed.path.split("/") if p])

    # 14. File extension length
    if "." in parsed.path:
        ext = parsed.path.split(".")[-1]
        features["this.fileExtLen"] = len(ext)
    else:
        features["this.fileExtLen"] = 0

    return features
