#include <TCanvas.h>
#include <TH1F.h>
#include <TFile.h>
#include <TLegend.h>
#include <TStyle.h>

void plotTPCChiComparison()
{
  // Open the file containing the histograms
  TFile *file = TFile::Open("AnalysisResults.root");

  // Navigate to the directory where the histograms are stored
  file->cd("my-track-select");

  // Retrieve the TPC Chi histograms
  TH1F *TPCChiNoConstraints = (TH1F *)gDirectory->Get("TPCChiHistogram");
  TH1F *TPCChi035CrossedRows = (TH1F *)gDirectory->Get("035TPCChi");
  TH1F *TPCChi3570CrossedRows = (TH1F *)gDirectory->Get("3570TPCChi");
  TH1F *TPCChi70CrossedRows = (TH1F *)gDirectory->Get("70TPCChi");
  TH1F *EtaHistogram = (TH1F *)gDirectory->Get("EtaHistogram");

  // Check if histograms are found
  if (!TPCChiNoConstraints || !TPCChi035CrossedRows || !TPCChi3570CrossedRows || !TPCChi70CrossedRows || !EtaHistogram)
  {
    std::cerr << "Error: One or more TPC Chi histograms not found!" << std::endl;
    return;
  }

  // Normalize histograms

  /*   // Normalize histograms
    TPCChiNoConstraints->Scale(1.0 / TPCChiNoConstraints->Integral());
    TPCChi3570CrossedRows->Scale(1.0 / TPCChi3570CrossedRows->Integral());
    TPCChi70CrossedRows->Scale(1.0 / TPCChi70CrossedRows->Integral());
    TPCChi035CrossedRows->Scale(1.0 / TPCChi035CrossedRows->Integral());
   */
  // TPC Chi Comparison
  TCanvas *c1 = new TCanvas("c1", "TPC Chi Comparison", 800, 600);
  TPCChiNoConstraints->GetXaxis()->SetRangeUser(0, 3);

  // Set line colors for the histograms
  TPCChiNoConstraints->SetLineColor(kBlack);
  TPCChi035CrossedRows->SetLineColor(kBlue);
  TPCChi3570CrossedRows->SetLineColor(kGreen);
  TPCChi70CrossedRows->SetLineColor(kRed);
  EtaHistogram->SetLineColor(kMagenta);

  // Set line widths
  TPCChiNoConstraints->SetLineWidth(2);
  TPCChi035CrossedRows->SetLineWidth(2);
  TPCChi3570CrossedRows->SetLineWidth(2);
  TPCChi70CrossedRows->SetLineWidth(2);
  EtaHistogram->SetLineWidth(2);

  // Draw histograms on the same canvas
  TPCChiNoConstraints->Draw("HIST");
  TPCChi035CrossedRows->Draw("HIST SAME");
  TPCChi3570CrossedRows->Draw("HIST SAME");
  TPCChi70CrossedRows->Draw("HIST SAME");
  EtaHistogram->Draw("HIST SAME");

  // Add a legend for clarity
  TLegend *legend = new TLegend(0.7, 0.7, 0.9, 0.9);
  legend->AddEntry(TPCChiNoConstraints, "TPC Chi2 No Constraints", "l");
  legend->AddEntry(TPCChi035CrossedRows, "TPC Chi2 (0-35 Crossed Rows)", "l");
  legend->AddEntry(TPCChi3570CrossedRows, "TPC Chi2 (35-70 Crossed Rows)", "l");
  legend->AddEntry(TPCChi70CrossedRows, "TPC Chi2 (70+ Crossed Rows)", "l");
  legend->AddEntry(EtaHistogram, "Track Select", "l");
  legend->Draw();

  // Set title and axis labels
  TPCChiNoConstraints->SetTitle("Comparison of TPC Chi2 Histograms by Crossed Rows");
  TPCChiNoConstraints->GetXaxis()->SetTitle("Chi2 / cluster");
  TPCChiNoConstraints->GetYaxis()->SetTitle("Entries");

  // Save the canvas as a PNG file
  c1->SaveAs("Normalized_TPCChiComparison_CrossedRows.png");

  // Retrieve the ITS Chi histograms
  TH1F *NoHitITS = (TH1F *)gDirectory->Get("NoHitITS");
  TH1F *Hit13ITS = (TH1F *)gDirectory->Get("13HitITS");
  TH1F *AllHitITS = (TH1F *)gDirectory->Get("1AllHitITS");
  TH1F *H1370HitITS = (TH1F *)gDirectory->Get("H1370HitITS");
  TH1F *ITSChi = (TH1F *)gDirectory->Get("ITSChi"); // New histogram

  // Check if histograms are found
  if (!NoHitITS || !Hit13ITS || !AllHitITS || !ITSChi)
  {
    std::cerr << "Error: One or more ITS Chi histograms not found!" << std::endl;
    return;
  }

  // ITS Chi Comparison
  TCanvas *c2 = new TCanvas("c2", "ITS Chi Comparison", 800, 600);

  // Set line colors for the histograms
  NoHitITS->SetLineColor(kBlack);
  Hit13ITS->SetLineColor(kBlue);
  AllHitITS->SetLineColor(kRed);
  H1370HitITS->SetLineColor(kGreen);
  ITSChi->SetLineColor(kMagenta); // New histogram color

  // Set line widths
  NoHitITS->SetLineWidth(2);
  Hit13ITS->SetLineWidth(2);
  AllHitITS->SetLineWidth(2);
  H1370HitITS->SetLineWidth(2);
  ITSChi->SetLineWidth(2); // New histogram line width

  // Draw histograms on the same canvas
  Hit13ITS->Draw("HIST ");
  ITSChi->Draw("HIST SAME"); // Draw new histogram
  AllHitITS->Draw("HIST SAME");
  NoHitITS->Draw("HIST SAME");
  H1370HitITS->Draw("HIST SAME");
  ITSChi->Draw("HIST SAME"); // Draw new histogram

  // Add a legend for clarity
  TLegend *legend2 = new TLegend(0.7, 0.7, 0.9, 0.9);
  legend2->AddEntry(NoHitITS, "No hit in layer 0, NCls >= 3", "l");
  legend2->AddEntry(Hit13ITS, "Hit in every IB layer", "l");
  legend2->AddEntry(AllHitITS, "Hit in layer 0", "l");
  legend2->AddEntry(H1370HitITS, "Hit in layer 0, NCls >= 3, TPC CR > 70", "l");
  legend2->AddEntry(ITSChi, "Track Select", "l"); // New legend entry
  legend2->Draw();

  // Set title and axis labels
  ITSChi->SetTitle("Comparison of ITS Chi2 Histograms by Cluster Size");
  ITSChi->GetXaxis()->SetTitle("Chi2 / cluster");
  ITSChi->GetYaxis()->SetTitle("Entries");

  // Save the canvas as a PNG file
  c2->SaveAs("Normalized_ITSChiComparison.png");

  // Retrieve the ITS Chi histograms
  TH1F *NoHitITSDCA = (TH1F *)gDirectory->Get("NoHitITSDCA");
  TH1F *Hit13ITSDCA = (TH1F *)gDirectory->Get("13HitITSDCA");
  TH1F *AllHitITSDCA = (TH1F *)gDirectory->Get("1AllHitITSDCA");
  TH1F *H1370HitITSDCA = (TH1F *)gDirectory->Get("H1370HitITSDCA");
  TH1F *DCAHistogram = (TH1F *)gDirectory->Get("DCAHistogram");

  // Check if histograms are found
  if (!NoHitITS || !Hit13ITS || !AllHitITS || !DCAHistogram)
  {
    std::cerr << "Error: One or more ITS Chi histograms not found!" << std::endl;
    return;
  }

  // ITS Chi Comparison
  TCanvas *c3 = new TCanvas("c3", "DCA Comparison", 800, 600);
  c3->SetLogy();

  // Set line colors for the histograms
  NoHitITSDCA->SetLineColor(kBlack);
  Hit13ITSDCA->SetLineColor(kBlue);
  AllHitITSDCA->SetLineColor(kRed);
  H1370HitITSDCA->SetLineColor(kGreen);
  DCAHistogram->SetLineColor(kMagenta);

  // Set line widths
  NoHitITSDCA->SetLineWidth(2);
  Hit13ITSDCA->SetLineWidth(2);
  AllHitITSDCA->SetLineWidth(2);
  H1370HitITSDCA->SetLineWidth(2);
  DCAHistogram->SetLineWidth(2);

  // Draw histograms on the same canvas
  AllHitITSDCA->Draw("HIST");
  Hit13ITSDCA->Draw("HIST SAME");
  NoHitITSDCA->Draw("HIST SAME");
  H1370HitITSDCA->Draw("HIST SAME");
  DCAHistogram->Draw("HIST SAME");

  // Add a legend for clarity
  TLegend *legend3 = new TLegend(0.7, 0.7, 0.9, 0.9);
  legend3->AddEntry(NoHitITS, "No hit in layer 0, NCls >= 3", "l");
  legend3->AddEntry(Hit13ITS, "it in every IB layer", "l");
  legend3->AddEntry(AllHitITS, "Hit in layer 0", "l");
  legend3->AddEntry(H1370HitITS, "Hit in layer 0, NCls >= 3, TPC CR > 70", "l");
  legend3->AddEntry(DCAHistogram, "Track Select", "l");

  legend3->Draw();

  // Set title and axis labels
  AllHitITSDCA->SetTitle("Comparison of DCA Histograms");
  AllHitITSDCA->GetXaxis()->SetTitle("dcaXY");
  AllHitITSDCA->GetYaxis()->SetTitle(" Entries");

  // Save the canvas as a PNG file
  c3->SaveAs("DCAComparison.png");

  // Retrieve the ITS Chi histograms
  TH1F *NoHitITSPT = (TH1F *)gDirectory->Get("NoHitITSPT");
  TH1F *Hit13ITSPT = (TH1F *)gDirectory->Get("13HitITSPT");
  TH1F *AllHitITSPT = (TH1F *)gDirectory->Get("1AllHitITSPT");
  TH1F *H1370HitITSPT = (TH1F *)gDirectory->Get("H1370HitITSPT");
  TH1F *PtHistogram = (TH1F *)gDirectory->Get("PtHistogram"); // New histogram

  // Check if histograms are found
  if (!NoHitITSPT || !Hit13ITSPT || !AllHitITSPT || !PtHistogram)
  {
    std::cerr << "Error: One or more ITS Chi histograms not found!" << std::endl;
    return;
  }

  // ITS Chi Comparison
  TCanvas *c4 = new TCanvas("c4", "PT Comparison", 800, 600);

  // Set line colors for the histograms
  NoHitITSPT->SetLineColor(kBlack);
  Hit13ITSPT->SetLineColor(kBlue);
  AllHitITSPT->SetLineColor(kRed);
  H1370HitITSPT->SetLineColor(kGreen);
  PtHistogram->SetLineColor(kMagenta); // New histogram color

  // Set line widths
  NoHitITSPT->SetLineWidth(2);
  Hit13ITSPT->SetLineWidth(2);
  AllHitITSPT->SetLineWidth(2);
  H1370HitITSPT->SetLineWidth(2);
  PtHistogram->SetLineWidth(2); // New histogram line width

  // Draw histograms on the same canvas
  Hit13ITSPT->Draw("HIST");
  AllHitITSPT->Draw("HIST SAME");
  PtHistogram->Draw("HIST SAME"); // Draw new histogram 
  NoHitITSPT->Draw("HIST SAME");
  H1370HitITSPT->Draw("HIST SAME");


  // Add a legend for clarity
  TLegend *legend4 = new TLegend(0.7, 0.7, 0.9, 0.9);
  legend4->AddEntry(NoHitITSPT, "No hit in layer 0, NCls >= 3", "l");
  legend4->AddEntry(Hit13ITSPT, "Hit in every IB layer", "l");
  legend4->AddEntry(AllHitITSPT, "Hit in layer 0", "l");
  legend4->AddEntry(H1370HitITSPT, "Hit in layer 0, NCls >= 3, TPC CR > 70", "l");
  legend4->AddEntry(PtHistogram, "Track Select", "l"); // New legend entry

  legend4->Draw();

  // Set title and axis labels
  AllHitITSPT->SetTitle("Comparison of PT Histograms");
  AllHitITSPT->GetXaxis()->SetTitle("pT");
  AllHitITSPT->GetYaxis()->SetTitle("Entries");

  // Save the canvas as a PNG file
  c4->SaveAs("PTComparison.png");
}
