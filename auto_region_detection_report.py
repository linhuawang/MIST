import joblib
import os
import ReST
import matplotlib.backends.backend_pdf

def generate_process_report(input_folder, output_folder = 'none',
    ReST_args = {"species":"Mouse", "min_sim": 0.1, 
    "min_region":40, "gap":0.02,
     "sigma":0.5, "region_min":2,
     "hvg_prop":  0.8, "n_pcs": 30}):

    pdf = matplotlib.backends.backend_pdf.PdfPages(f"{output_folder}/region_detection_report.pdf")
    rd = ReST.ReST(input_folder)
    rd.preprocess(species=ReST_args['species'], hvg_prop=ReST_args['hvg_prop'], n_pcs=ReST_args['n_pcs'])

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    rd.extract_regions(min_sim = ReST_args['min_sim'], 
                        min_region = ReST_args['min_region'],
                        gap  = ReST_args['gap'],
                        sigma = ReST_args['sigma'],
                        region_min = ReST_args['region_min'])

    pdf.savefig(rd.thr_opt_fig)

    # Plot region boundaries
    rd.assign_region_colors() ## assign a color to each region by default
    f0 = rd.plot_region_boundaries(by='UMI')
    pdf.savefig(f0)

    # plot regional markers
    rd.extract_regional_markers(mode='all')
    f1 = rd.plot_region_volcano()
    pdf.savefig(f1)

    # plot gsea results
    try:
        rd.runGSEA(mode='all', species=rd.species)
        f2 = rd.plot_region_enrichment(top=3)
        pdf.savefig(f2)
    except:
        pass

    pdf.close()
    rd.save(output_folder)
    # joblib.dump(rd, f'{output_folder}/ReST.job')
    # #rd.adata.write(f'{output_folder}/adata.h5')
    # region_deg_df = rd.region_deg_results
    # region_deg_df.to_csv(f"{output_folder}/regional_marker_df.csv")