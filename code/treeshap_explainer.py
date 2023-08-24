import shap
import config

def get_treeshap_explanation(x_train, x_test, top_instance_loc_list, bottom_instance_loc_list, model, model_name):
    
    # Define the KernelSHAP explainer
    explainer_tree = shap.TreeExplainer(model=model, data=x_train, model_output="raw")

    for instance_loc in top_instance_loc_list:
        # Select instance
        instance = x_test.iloc[instance_loc]
        # Find its Project ID
        project_id = instance["Project ID"]
        # Drop the Project ID value from the instance since its not a feature
        instance = instance.drop(["Project ID"])
        # Find the explanation
        treeshap_values = explainer_tree.shap_values(instance)

        # Visualize and save
        filepath = f'{config.TREESHAP_DEST}top/treeshap_exp_{project_id}_{model_name}.png'
        shap.force_plot(explainer_tree.expected_value[0], 
                    treeshap_values[0],
                    instance,
                    show=False, 
                    matplotlib=True, 
                    text_rotation=45).savefig(filepath, format = "png", dpi = 150, bbox_inches = 'tight')
        
    
    for instance_loc in bottom_instance_loc_list:
        # Select instance
        instance = x_test.iloc[instance_loc]
        # Find its Project ID
        project_id = instance["Project ID"]
        # Drop the Project ID value from the instance since its not a feature
        instance = instance.drop(["Project ID"])
        # Find the explanation
        treeshap_values = explainer_tree.shap_values(instance)

        # Visualize and save
        filepath = f'{config.TREESHAP_DEST}bottom/treeshap_exp_{project_id}_{model_name}.png'
        shap.force_plot(explainer_tree.expected_value[0], 
                    treeshap_values[0],
                    instance,
                    show=False, 
                    matplotlib=True, 
                    text_rotation=45).savefig(filepath, format = "png", dpi = 150, bbox_inches = 'tight')
        
        