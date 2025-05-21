#!/usr/bin/env bash
set -e

mongo "mongodb://bantaim:tjiSamKAAww@192.168.1.101:27017/vcon-faq" --eval 'db.getSiblingDB("vcon-faq").vcons.deleteMany({}); db.getSiblingDB("vcon-faq").faqs.deleteMany({});'
source venv/bin/activate
python main.py 