import csv
import MySQLdb

mydb = MySQLdb.connect(host='localhost',
    user='root',
    passwd='P@ssw0rd',
    db='OMOP_CDM')
cursor = mydb.cursor()

csv_data = csv.reader(file('final.csv'))
for row in csv_data:

    cursor.execute('INSERT INTO note_nlp_new(note_nlp_id,note_id,section_concept_id,snippet,offset,lexical_variant,note_nlp_concept_id,nlp_system,nlp_date_time,term_exists) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', row)
    mydb.commit()

cursor.close()
print "Done"

