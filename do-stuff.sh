#!/bin/bash
rsync -avh --size-only -W /media/10900-hdd-0/old-thing/Freeswitch1/ /media/10900-hdd-0/Freeswitch1/
rm -rvf /media/10900-hdd-0/old-thing/Freeswitch1/
rsync -avh --size-only -W /media/10900-hdd-0/old-thing/Freeswitch2/ /media/10900-hdd-0/Freeswitch2/
rm -rvf /media/10900-hdd-0/old-thing/Freeswitch2/
rsync -avh --size-only -W /media/10900-hdd-0/to-sync/Freeswitch13/ /media/10900-hdd-0/Freeswitch13/
rm -rvf /media/10900-hdd-0/to-sync/Freeswitch13/
rsync -avh --size-only -W /media/10900-hdd-0/to-sync/Freeswitch14/ /media/10900-hdd-0/Freeswitch14/
rm -rvf /media/10900-hdd-0/to-sync/Freeswitch14/
