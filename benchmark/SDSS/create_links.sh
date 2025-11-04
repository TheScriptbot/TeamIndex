wget -q -O - https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/ \
| grep -oP 'calibObj-\d+-\d+-(gal|star)\.fits\.gz' \
| sort -u \
| sed 's|^|https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/|' \
> download-links.txt

