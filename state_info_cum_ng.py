import csv
import os
import sys
import subprocess
with open('with_line_num.txt','w') as wf:
	   fp=open("tot_info.c.gcov");
           j=1
	   for line in fp:
                        flag=line.split(":")[0];
			if "-" in flag:
                            wf.write(line)
                            continue
                        elif "#####" in flag:
                            wf.write(str(j)+""+line)
                            j=j+1
                        else:
                            wf.write(str(j)+""+line)
                            j=j+1
		
    


