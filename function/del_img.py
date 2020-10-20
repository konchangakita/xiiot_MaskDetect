import json
import logging

def main(ctx,msg):
    logging.info("***** Delete jpg image *****")
    rmsg = json.loads(msg)
    del rmsg['mask_img']

    logging.info("rmsg %s", rmsg)


    rmsg = json.dumps(rmsg).encode('utf-8')
    ctx.send(rmsg)