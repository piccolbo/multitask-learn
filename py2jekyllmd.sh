pweave -f pandoc multitask.py && cp multitask.md multitask.md.back && sed -e 's/(figures\//(\/assets\//' -e 's/)\\$/)/'< multitask.md.back > multitask.md  && rm -f multitask.md.back
