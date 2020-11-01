from crawl import testVideo






url = "https://api3-core-c-lq.amemv.com/aweme/v1/aweme/post/?source=0&max_cursor=0&sec_user_id=MS4wLjABAAAA4lUpdAc-9fhvZB4dJK-VEeKlldKpYmm6DqXVz-VYFcU&count=20&_rticket=1597333815790&oaid=6b3accdcb0a7f128&mcc_mnc=46011&ts=1597333815"
cookies = "install_id=4309687584954063; ttreq=1$e6154c8211d05b45df5a1d3fe79cfc0aff38cf8e; odin_tt=314f476d5a567bb6a98ca81897e543b122d578563fdd16e0441e8f20513e00c298148cd264ff98298dc88a3b7408c7304b91eb47df501da13ae515d6a9c7e768"
print(testVideo(url, cookies))
