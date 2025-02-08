#include <TCanvas.h>
#include <TH1F.h>
#include <TFile.h>
#include <TLegend.h>
#include <TStyle.h>

void TreeMacro()
{
    // Open the file containing the histograms
    TFile *file = TFile::Open("AnalysisResults.root");  // Replace with your actual file name
    
    // Navigate to your analysis directory
    file->cd("my-p-i-d-analysis");
    
    // Get the histograms
    TH2F *hTofExpSigvsP = (TH2F *)gDirectory->Get("TofExpSigvsP");
    TH2F *hTpcExpSigvsP = (TH2F *)gDirectory->Get("TpcExpSigvsP");

    // Get momentum histograms
    TH1F *hP_all = (TH1F *)gDirectory->Get("p_all");
    TH1F *hP_its = (TH1F *)gDirectory->Get("p_its");
    TH1F *hP_tpc = (TH1F *)gDirectory->Get("p_tpc");
    TH1F *hP_trd = (TH1F *)gDirectory->Get("p_trd");
    TH1F *hP_tof = (TH1F *)gDirectory->Get("p_tof");
    

    
    // Check if histograms are found
    if (!hTofExpSigvsP || !hTpcExpSigvsP)
    {
        std::cerr << "Error: Could not find the histograms" << std::endl;
        return;
    }
    
    // Create canvases
    TCanvas *c1 = new TCanvas("c1", "TOF Signal Ratio vs P", 800, 600);
    c1->SetGrid();  // Add grid
    c1->SetLogx();  // Set x-axis to logarithmic scale

    hTofExpSigvsP->Draw("colz");
    hTofExpSigvsP->SetTitle("TOF Signal Ratio vs P");
    hTofExpSigvsP->GetXaxis()->SetTitle("p (GeV/c)");
    hTofExpSigvsP->GetYaxis()->SetTitle("(t_{TOF} - t_{el}) / t_{el}");
    
    TCanvas *c2 = new TCanvas("c2", "TPC Signal Ratio vs P", 800, 600);
    c2->SetLogx();  // Set x-axis to logarithmic scale
    hTpcExpSigvsP->Draw("colz");
    c2->SetGrid();  // Add grid
    hTpcExpSigvsP->SetTitle("TPC Signal Ratio vs P");
    hTpcExpSigvsP->GetXaxis()->SetTitle("p (GeV/c)");
    hTpcExpSigvsP->GetYaxis()->SetTitle("(TPC dE/dx_{meas} - TPC dE/dx_{exp}^{e}) / TPC dE/dx_{exp}^{e}");


    // Third canvas for momentum comparison
    TCanvas *c3 = new TCanvas("c3", "Momentum Comparison", 800, 600);
    // c3->SetLogy();  // Set log scale for y-axis
    // c3->SetLogx();  // Set log scale for x-axis
    c3->SetGrid();
    
    // Normalize histograms
    hP_all->Scale(1.0/hP_all->Integral());
    hP_its->Scale(1.0/hP_its->Integral());
    hP_tpc->Scale(1.0/hP_tpc->Integral());
    hP_trd->Scale(1.0/hP_trd->Integral());
    hP_tof->Scale(1.0/hP_tof->Integral());
    
    // Set colors and styles
    hP_all->SetLineColor(kBlack);
    hP_its->SetLineColor(kRed);
    hP_tpc->SetLineColor(kBlue);
    hP_trd->SetLineColor(kGreen+2);
    hP_tof->SetLineColor(kMagenta);
    
    hP_all->SetLineWidth(2);
    hP_its->SetLineWidth(2);
    hP_tpc->SetLineWidth(2);
    hP_trd->SetLineWidth(2);
    hP_tof->SetLineWidth(2);
    
    // Draw histograms
    // hP_all->Draw("HIST");
    hP_its->Draw("HIST");
    hP_tpc->Draw("HIST SAME");
    hP_trd->Draw("HIST SAME");
    hP_tof->Draw("HIST SAME");
    
    // Create legend with entries
    TLegend *leg = new TLegend(0.65, 0.65, 0.89, 0.89);
    // leg->AddEntry(hP_all, Form("All (N = %.0f)", hP_all->GetEntries()), "l");
    leg->AddEntry(hP_its, Form("ITS (N = %.0f)", hP_its->GetEntries()), "l");
    leg->AddEntry(hP_tpc, Form("TPC (N = %.0f)", hP_tpc->GetEntries()), "l");
    leg->AddEntry(hP_trd, Form("TRD (N = %.0f)", hP_trd->GetEntries()), "l");
    leg->AddEntry(hP_tof, Form("TOF (N = %.0f)", hP_tof->GetEntries()), "l");
    leg->SetBorderSize(0);
    leg->Draw();
    
    // Set titles
    hP_its->SetTitle("Momentum Comparison");
    hP_its->GetXaxis()->SetTitle("p (GeV/c)");
    hP_its->GetYaxis()->SetTitle("Normalized Counts");

     
}