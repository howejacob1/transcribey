import logging
import process



def start_thread(transcribed_vcons_queue):
    logging.info("Starting send results thread.")

def start_process(transcribed_vcons_queue, stats_queue):
    return process.start_process(target=start_thread, args=(transcribed_vcons_queue,))
