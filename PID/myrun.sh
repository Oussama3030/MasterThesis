export OPTIONS="-b --configuration json://myconfignew.json --aod-writer-resfile myTrees --aod-writer-ntfmerge 100000 --aod-writer-keep AOD/TPCDATA/0:TpcData"

o2-analysis-timestamp ${OPTIONS} | o2-analysis-trackselection ${OPTIONS} | o2-analysis-track-propagation ${OPTIONS} | o2-analysis-pid-tof-base ${OPTIONS} | o2-analysis-pid-tof-beta ${OPTIONS} | o2-analysis-event-selection ${OPTIONS} | o2-analysis-pid-tof-full ${OPTIONS} | o2-analysis-pid-tpc ${OPTIONS} | o2-analysis-pid-tpc-base ${OPTIONS} | o2-analysis-em-my-pid-analysis ${OPTIONS} 

