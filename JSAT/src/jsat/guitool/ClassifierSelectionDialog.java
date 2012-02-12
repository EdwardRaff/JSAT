
package jsat.guitool;

import java.awt.BorderLayout;
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
import jsat.classifiers.bayesian.MultivariateNormals;
import jsat.classifiers.bayesian.NaiveBayes;
import jsat.classifiers.boosting.AdaBoostM1;
import jsat.classifiers.boosting.SAMME;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.classifiers.trees.DecisionStump;
import jsat.classifiers.trees.DecisionTree;
import jsat.classifiers.trees.ID3;
import jsat.classifiers.trees.RandomForest;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.vectorcollection.VPTree.VPTreeFactory;

/**
 *
 * @author Edward Raff
 */
public class ClassifierSelectionDialog extends JDialog
{
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
        abstract public String toString();
        
        abstract public Classifier getNewClassifier();
        
    }
    
    private static final List<ClassifierInfo> possibleClassifiers = new ArrayList<ClassifierSelectionDialog.ClassifierInfo>()
    {{
        add(new ClassifierInfo() {

                @Override
                public String toString()
                {
                    return "Decision Stump";
                }

                @Override
                public Classifier getNewClassifier()
                {
                    return new DecisionStump();
                }
            });
        add(new ClassifierInfo() {

                @Override
                public String toString()
                {
                    return "Decision Tree";
                }

                @Override
                public Classifier getNewClassifier()
                {
                    return new DecisionTree();
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public boolean canTrain(ClassificationDataSet cds)
                {
                    return cds.getNumNumericalVars() == 0;
                }

                @Override
                public String toString()
                {
                    return "ID3";
                }

                @Override
                public Classifier getNewClassifier()
                {
                    return new DecisionTree();
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public String toString()
                {
                    return "Random Forest [200]";
                }

                @Override
                public Classifier getNewClassifier()
                {
                    return new RandomForest(200);
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public String toString()
                {
                    return "Boosted Decision Trees (Depth: 4, SAMME [50])";
                }

                @Override
                public Classifier getNewClassifier()
                {
                    return new SAMME(new DecisionTree(4, 5, DecisionTree.PruningMethod.NONE, 0.1), 50);
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public boolean canTrain(ClassificationDataSet cds)
                {
                    return cds.getPredicting().getNumOfCategories() == 2;
                }

                @Override
                public String toString()
                {
                    return "AdaBoostM1(Decision Stumps)";
                }

                @Override
                public Classifier getNewClassifier()
                {
                    return new AdaBoostM1(new DecisionStump(), 200);
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public String toString()
                {
                    return "Naive Bayes";
                }

                @Override
                public Classifier getNewClassifier()
                {
                    return new NaiveBayes();
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public boolean canTrain(ClassificationDataSet cds)
                {
                    return cds.getNumCategoricalVars() == 0;
                }
            
                @Override
                public String toString()
                {
                    return "Multivariate Normals";
                }

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
                public String toString()
                {
                    return "Nearest Neighbour";
                }

                @Override
                public Classifier getNewClassifier()
                {
                    return new NearestNeighbour(1, new VPTreeFactory<VecPaired<Double, Vec>>());
                }
            });
        
        add(new ClassifierInfo() {

                @Override
                public boolean canTrain(ClassificationDataSet cds)
                {
                    return cds.getNumCategoricalVars() == 0;
                }
            
                @Override
                public String toString()
                {
                    return "5-Nearest Neighbours";
                }

                @Override
                public Classifier getNewClassifier()
                {
                    return new NearestNeighbour(5, new VPTreeFactory<VecPaired<Double, Vec>>());
                }
            });
    }};
    
    private ClassificationDataSet dataSet;
    private List<JCheckBox> checkBoxes;
    private boolean hitCancel = false;

    public ClassifierSelectionDialog(ClassificationDataSet dataSet, Frame owner)
    {
        super(owner, "Title", true);
        this.dataSet = dataSet;
        setLayout(new BorderLayout());
        JPanel jPanel = new JPanel(new GridLayout(possibleClassifiers.size(), 1));
        checkBoxes = new ArrayList<JCheckBox>();
        for(ClassifierInfo cinf : possibleClassifiers)
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

            public void actionPerformed(ActionEvent e)
            {
                setVisible(false);
            }
        });
        
        jPanel.add(jButton);
        
        jButton = new JButton("Cancel");
        jButton.addActionListener(new ActionListener() {

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
    
    public List<Classifier> getSelectedClassifiers()
    {
        List<Classifier> classifiers = new ArrayList<Classifier>();
        
        for(int i = 0; i < possibleClassifiers.size(); i++)
        {
            if(checkBoxes.get(i).isSelected())
                classifiers.add(possibleClassifiers.get(i).getNewClassifier());
        }
        
        return classifiers;
    }
    
}
