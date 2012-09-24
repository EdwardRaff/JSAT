
package jsat.guitool;

import java.awt.BorderLayout;
import java.awt.Dialog;
import java.awt.FlowLayout;
import java.awt.Frame;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JDialog;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.MultinomialLogisticRegression;
import jsat.classifiers.bayesian.BestClassDistribution;
import jsat.classifiers.bayesian.MultivariateNormals;
import jsat.classifiers.bayesian.NaiveBayes;
import jsat.classifiers.boosting.AdaBoostM1;
import jsat.classifiers.boosting.LogitBoost;
import jsat.classifiers.boosting.SAMME;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.classifiers.neuralnetwork.BackPropagationNet;
import jsat.classifiers.neuralnetwork.SOM;
import jsat.classifiers.trees.DecisionStump;
import jsat.classifiers.trees.DecisionTree;
import jsat.classifiers.trees.RandomForest;
import jsat.distributions.multivariate.MetricKDE;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.vectorcollection.VPTree.VPTreeFactory;
import jsat.parameters.Parameterized;

/**
 * Simple code for providing a dialog that allows for the selection of multiple classifeirs
 * @author Edward Raff
 */
public class ClassifierSelectionDialog extends JDialog
{
    private Frame owner;
    private List<ClassifierInfo> listInUse;
    //Provide a little wrapper class to use and filter classifier options
    private static abstract class ClassifierInfo 
    {
        
        /**
         * Returns true if the classifier could be trained on the given data set. False if it could not
         * @param cds the data set
         * @return true if it could be fit. 
         */
        public boolean canTrain(ClassificationDataSet cds)//Over ride only if there are data set restrictions
        {
            return true;
        }

        @Override
        public String toString()
        {
            return getNewClassifier().getClass().getSimpleName();
        }
        
        abstract public Classifier getNewClassifier();
        
    }
    
    private static abstract class ClassifierInfoWeakLearner extends ClassifierInfo
    {
        abstract public Classifier getNewClassifier(Classifier weakLearner);

        @Override
        public Classifier getNewClassifier()
        {
            return getNewClassifier(new DecisionStump());
        }
    }
    
    private static final List<ClassifierInfo> weakClassifiers = new ArrayList<ClassifierSelectionDialog.ClassifierInfo>()
    {{
        add(new ClassifierInfo() {

                @Override
                public Classifier getNewClassifier()
                {
                    return new DecisionStump();
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public Classifier getNewClassifier()
                {
                    return new DecisionTree();
                }
            });
    }};
    
    private static final List<ClassifierInfo> possClass = new ArrayList<ClassifierSelectionDialog.ClassifierInfo>()
    {{
        addAll(weakClassifiers);
        
        add(new ClassifierInfo() {
                @Override
                public Classifier getNewClassifier()
                {
                    return new RandomForest(200);
                }
            });
        
        add(new ClassifierInfoWeakLearner() {
            
                @Override
                public Classifier getNewClassifier(Classifier weakLearner)
                {
                    return new AdaBoostM1(weakLearner, 200);
                }
            });
        
        add(new ClassifierInfoWeakLearner() {

                @Override
                public Classifier getNewClassifier(Classifier weakLearner)
                {
                    return new SAMME(weakLearner, 100);
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public Classifier getNewClassifier()
                {
                    return new NaiveBayes();
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public Classifier getNewClassifier()
                {
                    return new MultivariateNormals(true);
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public boolean canTrain(ClassificationDataSet cds)
                {
                    return cds.getNumCategoricalVars() == 0;
                }

                @Override
                public Classifier getNewClassifier()
                {
                    return new NearestNeighbour(1);
                }
            });
        
        add(new ClassifierInfo() {

            @Override
            public String toString()
            {
                return "Kernel Density Estimator";
            }
            
            @Override
            public Classifier getNewClassifier()
            {
                return new BestClassDistribution(new MetricKDE());
            }
        });
        
        add(new ClassifierInfo() 
        {

                @Override
                public boolean canTrain(ClassificationDataSet cds)
                {
                    return cds.getNumCategoricalVars() == 0;
                }
                
                @Override
                public Classifier getNewClassifier()
                {
                    return new MultinomialLogisticRegression();
                }
         });
        
        add(new ClassifierInfo() {

            @Override
            public Classifier getNewClassifier()
            {
                return new SOM(5, 5);
            }
        });
    }};
    
    private ClassificationDataSet dataSet;
    private List<JCheckBox> checkBoxes;
    private boolean hitCancel = false;
    
    private List<Classifier> pClassifiers = new ArrayList<Classifier>()
    {{
        add(new DecisionStump());
    }};

    public ClassifierSelectionDialog(ClassificationDataSet dataSet, Frame owner)
    {
        this(dataSet, owner, "Classifier Selection Dialog", false);
    }
    public ClassifierSelectionDialog(ClassificationDataSet dataSet, Frame owner, String title, boolean weakOnly)
    {
        super(owner, title, true);
        this.owner = owner;
        this.dataSet = dataSet;
        setLayout(new BorderLayout());
        JPanel jPanel = new JPanel(new GridLayout(possClass.size(), 1));
        checkBoxes = new ArrayList<JCheckBox>();
        listInUse = weakOnly ? weakClassifiers : possClass;
        for(ClassifierInfo cinf : listInUse )
        {
            JCheckBox checkBox = new JCheckBox(cinf.toString());
            checkBox.setEnabled(cinf.canTrain(dataSet));
            jPanel.add(checkBox);
            checkBoxes.add(checkBox);
        }
        
        add(new JScrollPane(jPanel), BorderLayout.CENTER);
        
        jPanel = new JPanel(new FlowLayout());
        JButton jButton = new JButton("Ok");
        jButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e)
            {
                setVisible(false);
            }
        });
        
        jPanel.add(jButton);
        
        jButton = new JButton("Cancel");
        jButton.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e)
            {
                hitCancel = true;
                setVisible(false);
            }
        });
        jPanel.add(jButton);
        add(jPanel, BorderLayout.SOUTH);
    }

    public boolean isCanceled()
    {
        return hitCancel;
    }
    
    
    private List<String> names;
    
    public List<Classifier> getSelectedClassifiers()
    {
        names = new ArrayList<String>();
        List<Classifier> classifiers = new ArrayList<Classifier>();
        
        for(int i = 0; i < listInUse.size(); i++)
        {
            if(checkBoxes.get(i).isSelected())
            {
                
                if(possClass.get(i) instanceof ClassifierInfoWeakLearner)
                {
                    ClassifierInfoWeakLearner ciwl = (ClassifierInfoWeakLearner) possClass.get(i);
                    ClassifierSelectionDialog weakSelect;
                    weakSelect = new ClassifierSelectionDialog(dataSet, owner, 
                            "Select weak learner for " + ciwl.toString(), true);
                    
                    weakSelect.setSize(400, 400);
                    weakSelect.setVisible(true);

                    if(weakSelect.isCanceled())
                        continue;
                    List<Classifier> selected = weakSelect.getSelectedClassifiers();
                    List<String> selectedName = weakSelect.getSelectedNames();
                    for(int z = 0; z < selected.size(); z++)
                    {
                        Classifier weak = selected.get(z);
                        Classifier finalClass = ciwl.getNewClassifier(weak);
                        classifiers.add(finalClass);
                        names.add(ciwl.toString() + " using " + selectedName.get(z));
                        if(finalClass instanceof Parameterized)
                            ParameterPanel.showParameterDiag(owner, "Select Parameters for " + names.get(names.size()-1), (Parameterized)finalClass);
                    }
                }
                else
                {
                    Classifier finalClass = possClass.get(i).getNewClassifier();
                    classifiers.add(finalClass);
                    names.add(possClass.get(i).toString());
                    if(finalClass instanceof Parameterized)
                            ParameterPanel.showParameterDiag(owner, "Select Parameters for " + names.get(names.size()-1), (Parameterized)finalClass);
                }
            }
        }
        
        
        
        return classifiers;
    }
    
    /**
     * Returns more descriptive names for the classifiers in the same order as 
     * they are returned by {@link #getSelectedClassifiers() }
     * @return more descriptive names for each classifier
     */
    public List<String> getSelectedNames()
    {
        return names;
    }
    
}
