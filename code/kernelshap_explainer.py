import shap
import config

def get_shap_explanation(x_train, x_test, top_instance_loc_list, bottom_instance_loc_list, model, model_name):
    
    # Define the KernelSHAP explainer
    explainer_shap = shap.KernelExplainer(model=model.predict_proba, data=x_train)


    for instance_loc in top_instance_loc_list:
        # Explain for the selected instance
        instance = x_test.iloc[instance_loc]
        shap_values = explainer_shap.shap_values(instance)

        # Visualize and save
        # Can be either 0 or 1 for binary classification
        #shap.force_plot(explainer_shap.expected_value[0], shap_values[0], instance)
        filepath = f'{config.SHAP_DEST}top/shap_exp_{instance_loc}_{model_name}.png'
        shap.force_plot(explainer_shap.expected_value[0], 
                        shap_values[0], 
                        instance, 
                        show=False, 
                        matplotlib=True, 
                        text_rotation=45).savefig(filepath, format = "png", dpi = 150, bbox_inches = 'tight') 
        
    for instance_loc in bottom_instance_loc_list:
        # Explain for the selected instance
        instance = x_test.iloc[instance_loc]
        shap_values = explainer_shap.shap_values(instance)

        # Visualize and save
        # Can be either 0 or 1 for binary classification
        #shap.force_plot(explainer_shap.expected_value[0], shap_values[0], instance)
        filepath = f'{config.SHAP_DEST}bottom/shap_exp_{instance_loc}_{model_name}.png'
        shap.force_plot(explainer_shap.expected_value[0], 
                        shap_values[0], 
                        instance, 
                        show=False, 
                        matplotlib=True, 
                        text_rotation=45).savefig(filepath, format = "png", dpi = 150, bbox_inches = 'tight') 

    