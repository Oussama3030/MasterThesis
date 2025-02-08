#include <TCanvas.h>
#include <TH1F.h>
#include <TFile.h>
#include <TLegend.h>
#include <TStyle.h>

void plotITSChi2Comparison() {
  // Open the file containing the histograms
  TFile* file = TFile::Open("AnalysisResults.root");
  
  // Navigate to the directory where the histograms are stored
  file->cd("my-track-task");

  // Retrieve the ITS histograms
  TH1F* itsChi2Histogram = (TH1F*)gDirectory->Get("itsChi2Histogram");
  TH1F* itsChiNoLayer0 = (TH1F*)gDirectory->Get("itsChiNoLayer0");
  TH1F* itsChiLayer0 = (TH1F*)gDirectory->Get("itsChiLayer0");
  TH1F* itsNIBChi = (TH1F*)gDirectory->Get("itsNIBChi");
  TH1F* itsChiTPC = (TH1F*)gDirectory->Get("itsChiTPC");
  TH1F* itsN7TPCChi = (TH1F*)gDirectory->Get("itsN7TPCChi");
  TH1F* itsChiTrack = (TH1F*)gDirectory->Get("itsChiTrack");


  if (!itsChi2Histogram || !itsChiNoLayer0 || !itsChiLayer0 || !itsNIBChi || !itsChiTPC || !itsN7TPCChi) {
    std::cerr << "Error: One or more ITS histograms not found!" << std::endl;
    return;
  }
/* 
  // Normalize histograms
  itsChi2Histogram->Scale(1.0 / itsChi2Histogram->Integral());
  itsChiNoLayer0->Scale(1.0 / itsChiNoLayer0->Integral());
  itsChiLayer0->Scale(1.0 / itsChiLayer0->Integral());
  itsNIBChi->Scale(1.0 / itsNIBChi->Integral());
  itsChiTPC->Scale(1.0 / itsChiTPC->Integral());
  itsN7TPCChi->Scale(1.0 / itsN7TPCChi->Integral());
 */
  // ITS Chi2 Comparison
  TCanvas* c1 = new TCanvas("c1", "ITS Chi2 Comparison", 800, 600);
  
  itsChi2Histogram->SetLineColor(kRed);
  itsChiNoLayer0->SetLineColor(kOrange);
  itsChiLayer0->SetLineColor(kBlue);
  itsNIBChi->SetLineColor(kBlack);
  itsChiTPC->SetLineColor(kGreen);
  itsN7TPCChi->SetLineColor(kMagenta);
  itsChiTrack->SetLineColor(kCyan);


  itsChi2Histogram->SetLineWidth(2);
  itsChiNoLayer0->SetLineWidth(2);
  itsChiLayer0->SetLineWidth(2);
  itsNIBChi->SetLineWidth(2);
  itsChiTPC->SetLineWidth(2);
  itsN7TPCChi->SetLineWidth(2);
  itsChiTrack->SetLineWidth(2);

  itsChi2Histogram->Draw("HIST");
  itsChiLayer0->Draw("HIST SAME");
  itsChiNoLayer0->Draw("HIST SAME");
  itsNIBChi->Draw("HIST SAME");
  itsChiTPC->Draw("HIST SAME");
  itsN7TPCChi->Draw("HIST SAME");
  itsChiTrack->Draw("HIST SAME");

  TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
  legend->AddEntry(itsChi2Histogram, "ITS", "l");
  legend->AddEntry(itsNIBChi, "ITS NIB", "l");
  legend->AddEntry(itsChiNoLayer0, "ITS No Layer 0", "l");
  legend->AddEntry(itsChiLayer0, "ITS Included Layer 0", "l");
  legend->AddEntry(itsChiTPC, "ITS Good TPC", "l");
  legend->AddEntry(itsN7TPCChi, "ITS 7 Hits + Good TPC", "l");
  legend->AddEntry(itsChiTrack, "Track Selection O2", "l");


  legend->Draw();
  
  itsChi2Histogram->SetTitle("Comparison of Normalized ITS Chi2 Histograms");
  itsChi2Histogram->GetXaxis()->SetTitle("Chi2 / cluster");
  itsChi2Histogram->GetYaxis()->SetTitle("Entries");

  c1->SaveAs("Normalized_ITSChi2Comparison.png");


  // TPC NCl Comparison
  TH1F* tpcNClHistogram = (TH1F*)gDirectory->Get("tpcNClHistogram");
  TH1F* tpcNCl0Histogram = (TH1F*)gDirectory->Get("tpcNCl0Histogram");

  if (!tpcNClHistogram || !tpcNCl0Histogram) {
    std::cerr << "Error: One or more TPC NCl histograms not found!" << std::endl;
    return;
  }

  TCanvas* c2 = new TCanvas("c2", "TPC NCl Comparison", 800, 600);
  
  tpcNClHistogram->SetLineColor(kMagenta);
  tpcNCl0Histogram->SetLineColor(kCyan);
  
  tpcNClHistogram->SetLineWidth(2);
  tpcNCl0Histogram->SetLineWidth(2);
  
  tpcNClHistogram->Draw("HIST");
  tpcNCl0Histogram->Draw("HIST SAME");
  
  TLegend* legend2 = new TLegend(0.7, 0.7, 0.9, 0.9);
  legend2->AddEntry(tpcNClHistogram, "TPC NCl", "l");
  legend2->AddEntry(tpcNCl0Histogram, "TPC NCl0", "l");
  legend2->Draw();
  
  tpcNClHistogram->SetTitle("Comparison of TPC NCl Histograms");
  tpcNClHistogram->GetXaxis()->SetTitle("TPC Clusters");
  tpcNClHistogram->GetYaxis()->SetTitle("Entries");
  
  c2->SaveAs("TPCNClComparison.png");

  // TPC Chi2 Comparison
  TH1F* tpcChi2Histogram = (TH1F*)gDirectory->Get("tpcChi2Histogram");
  TH1F* tpc0Chi2Histogram = (TH1F*)gDirectory->Get("tpc0Chi2Histogram");

  if (!tpcChi2Histogram || !tpc0Chi2Histogram) {
    std::cerr << "Error: One or more TPC Chi2 histograms not found!" << std::endl;
    return;
  }

  TCanvas* c3 = new TCanvas("c3", "TPC Chi2 Comparison", 800, 600);
  
  tpcChi2Histogram->SetLineColor(kOrange);
  tpc0Chi2Histogram->SetLineColor(kViolet);
  
  tpcChi2Histogram->SetLineWidth(2);
  tpc0Chi2Histogram->SetLineWidth(2);
  
  tpcChi2Histogram->Draw("HIST");
  tpc0Chi2Histogram->Draw("HIST SAME");
  
  TLegend* legend3 = new TLegend(0.7, 0.7, 0.9, 0.9);
  legend3->AddEntry(tpcChi2Histogram, "TPC Chi2", "l");
  legend3->AddEntry(tpc0Chi2Histogram, "TPC 0 Chi2", "l");
  legend3->Draw();
  
  tpcChi2Histogram->SetTitle("Comparison of TPC Chi2 Histograms");
  tpcChi2Histogram->GetXaxis()->SetTitle("Chi2 / cluster");
  tpcChi2Histogram->GetYaxis()->SetTitle("Entries");
  
  c3->SaveAs("TPCChi2Comparison.png");

/*   // DCA XY Comparison
  TH1F* dcaXYHistogram = (TH1F*)gDirectory->Get("dcaXYHistogram");
  TH1F* dcaXY1Histogram = (TH1F*)gDirectory->Get("dcaXY1Histogram");
  TH1F* dcaXYNIBHistogram = (TH1F*)gDirectory->Get("dcaXYNIBHistogram");

  if (!dcaXYHistogram || !dcaXYNIBHistogram) {
    std::cerr << "Error: One or more DCA XY histograms not found!" << std::endl;
    return;
  }

  TCanvas* c4 = new TCanvas("c4", "DCA XY Comparison", 800, 600);
  
  dcaXYHistogram->SetLineColor(kBlue);
  dcaXY1Histogram->SetLineColor(kGreen);
  dcaXYNIBHistogram->SetLineColor(kRed);
  
  dcaXYHistogram->SetLineWidth(2);
  dcaXY1Histogram->SetLineWidth(2);
  dcaXYNIBHistogram->SetLineWidth(2);
  
  dcaXYHistogram->Draw("HIST");
  dcaXYNIBHistogram->Draw("HIST SAME");
  dcaXY1Histogram->Draw("HIST SAME");

  
  TLegend* legend4 = new TLegend(0.7, 0.7, 0.9, 0.9);
  legend4->AddEntry(dcaXYHistogram, "DCA XY", "l");
  legend4->AddEntry(dcaXY1Histogram, "DCA XY No 1", "l");

  legend4->AddEntry(dcaXYNIBHistogram, "DCA XY NIB", "l");
  legend4->Draw();
  
  dcaXYHistogram->SetTitle("Comparison of DCA XY Histograms");
  dcaXYHistogram->GetXaxis()->SetTitle("DCA XY");
  dcaXYHistogram->GetYaxis()->SetTitle("Entries");
  
  c4->SaveAs("DCAXYComparison.png");
 */
  
// Additional DCA XY Histograms Comparison
TH1F* dcaXYHistogram = (TH1F*)gDirectory->Get("dcaXYHistogram");
TH1F* itsDCANoLayer0 = (TH1F*)gDirectory->Get("itsDCANoLayer0");
TH1F* itsDCALayer0 = (TH1F*)gDirectory->Get("itsDCALayer0");
TH1F* itsDCATPC = (TH1F*)gDirectory->Get("itsDCATPC");
TH1F* itsNIBDCA = (TH1F*)gDirectory->Get("itsNIBDCA");
TH1F* itsN7TPCDCA = (TH1F*)gDirectory->Get("itsN7TPCDCA");
TH1F* DCATrack = (TH1F*)gDirectory->Get("DCATrack");

if (!dcaXYHistogram || !itsDCANoLayer0 || !itsDCALayer0 || !itsDCATPC || !itsNIBDCA || !itsN7TPCDCA) {
    std::cerr << "Error: One or more new DCA histograms not found!" << std::endl;
    return;
}

TCanvas* c5 = new TCanvas("c5", "DCA XY Comparison", 800, 600);

// Set line colors for the new histograms
dcaXYHistogram->SetLineColor(kBlue);
itsDCANoLayer0->SetLineColor(kGreen);
itsDCALayer0->SetLineColor(kRed);
itsDCATPC->SetLineColor(kMagenta);
itsNIBDCA->SetLineColor(kCyan);
itsN7TPCDCA->SetLineColor(kOrange);
DCATrack->SetLineColor(kBlack);

// Set line widths
dcaXYHistogram->SetLineWidth(2);
itsDCANoLayer0->SetLineWidth(2);
itsDCALayer0->SetLineWidth(2);
itsDCATPC->SetLineWidth(2);
itsNIBDCA->SetLineWidth(2);
itsN7TPCDCA->SetLineWidth(2);
DCATrack->SetLineWidth(2);

// Draw the histograms on the same canvas
dcaXYHistogram->Draw("HIST");
itsDCANoLayer0->Draw("HIST SAME");
itsDCALayer0->Draw("HIST SAME");
itsDCATPC->Draw("HIST SAME");
itsNIBDCA->Draw("HIST SAME");
itsN7TPCDCA->Draw("HIST SAME");
DCATrack->Draw("HIST SAME");

// Add a legend for clarity
TLegend* legend5 = new TLegend(0.7, 0.7, 0.9, 0.9);
legend5->AddEntry(dcaXYHistogram, "DCA XY", "l");
legend5->AddEntry(itsDCANoLayer0, "ITS DCA No Layer 0", "l");
legend5->AddEntry(itsDCALayer0, "ITS DCA Layer 0", "l");
legend5->AddEntry(itsDCATPC, "ITS DCA TPC", "l");
legend5->AddEntry(itsNIBDCA, "ITS NIB DCA", "l");
legend5->AddEntry(itsN7TPCDCA, "ITS N7 TPC DCA", "l");
legend5->AddEntry(DCATrack, "Track Selection", "l");
legend5->Draw();

dcaXYHistogram->SetTitle("Comparison of New DCA XY Histograms");
dcaXYHistogram->GetXaxis()->SetTitle("DCA XY");
dcaXYHistogram->GetYaxis()->SetTitle("Entries");

// Save the canvas as a PNG file
c5->SaveAs("New_DCA_XY_Comparison.png");



  
}
