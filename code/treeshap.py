import shap
import config

def get_treeshap_explanation(x_train, x_test, instance_loc, model, model_name):

    #shap.initjs()
    
    # Define the KernelSHAP explainer
    explainer_tree = shap.TreeExplainer(model=model, data=x_train, model_output="raw")

    # Explain for the selected instance
    instance = x_test.iloc[instance_loc]
    treeshap_values = explainer_tree.shap_values(instance)

    # Visualize and save
    filepath = f'{config.TREESHAP_DEST}treeshap_exp_{instance_loc}_{model_name}.png'
    shap.force_plot(explainer_tree.expected_value[0], 
                treeshap_values[0],
                instance,
                show=False, 
                matplotlib=True, 
                text_rotation=45).savefig(filepath, format = "png", dpi = 150, bbox_inches = 'tight')
    
    return treeshap_values