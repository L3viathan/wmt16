Summary (first try with only pure linguistic features)
---------
It got over .92 accuracy in the classification task. The accuracy is quite fair since it was not trained to always predict the dominant class in data (two urls are not transaltion pairs of each other)

==========  =====   =====   =====
Set         top1    top5    top10
==========  =====   =====   =====
train       .070   .194     .188
validation  .045   .185     .139
Test        .039   .188     .145
==========  =====   =====   =====

The results was not high with the offical 1-1 recall of conference, but I could say it could work.

I could believe on, after I have better linguistic features and enrich them with heuristic features (url matching, shared words), the results would bebetter!

Impressivie results
------
*****------http://eu.blizzard.com/en-gb/community/blizzcast/archive/episode12.html
| --rs:in TOP5  
| --candidate_num:664 
| --gold(1):http://eu.blizzard.com/fr-fr/community/blizzcast/archive/episode12.html  
| ---top 10 cans: found only 3 potential  
| http://eu.blizzard.com/fr-fr/community/blizzcast/archive/episode6.html
| http://eu.blizzard.com/fr-fr/community/blizzcast/archive/episode12.html
| http://eu.blizzard.com/fr-fr/community/blizzcast/archive/episode3.htm

*****------http://cineuropa.mobi/interview.aspx?lang=en&documentID=109685
--rs:CORRECT
--can_num:15972
--gold(0):http://cineuropa.mobi/interview.aspx?lang=fr&documentID=109685
---top 10 cans:
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=109685
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=236696&messagecode=msgnolang
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=236696
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=250215
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=250215&messagecode=msgnolang
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=55408
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=54684



Wost domain in test set (probably in also validation and training set)
-----------
*****------http://cineuropa.mobi/interview.aspx?lang=en&documentID=64005
--rs:WRONG out of top10
--can_num:15972
--gold(19):http://cineuropa.mobi/interview.aspx?lang=fr&documentID=64005
---top 10 cans:
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=252168
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=77192
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=80530
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=145684
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=72049
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=108598
http://cineuropa.mobi/newsdetail.aspx?lang=fr&documentID=77063
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=81849
http://cineuropa.mobi/newsdetail.aspx?lang=fr&documentID=79717
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=63640

*****------http://cineuropa.mobi/interview.aspx?lang=en&documentID=146854
--rs:WRONG out of top10
--can_num:15972
--gold(360):http://cineuropa.mobi/interview.aspx?lang=fr&documentID=146854
---top 10 cans:
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=89835
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=235095
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=216445
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=259880
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=263533
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=286011
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=78091
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=84971
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=225575
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=259515

*****------http://cineuropa.mobi/interview.aspx?lang=en&documentID=60595
--rs:WRONG out of top10
--can_num:15972
--gold(73):http://cineuropa.mobi/interview.aspx?lang=fr&documentID=60595
---top 10 cans:
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=199966
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=239194
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=215590
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=287023
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=255638
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=31901
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=65207
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=258263
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=258875
http://cineuropa.mobi/interview.aspx?lang=fr&documentID=203082

For more details of all prediction: results/debug.txt
