package jsat.graphing;

import java.awt.*;
import java.util.*;
import java.util.List;
import jsat.classifiers.*;
import jsat.guitool.GUIUtils;
import jsat.utils.IndexTable;

/**
 * ROCPlot provides a simple means by which an Receiver Operating Characteristic
 * (ROC) curve can be ploted for one or more classifiers. <br>
 * ROC curves are a way of visually comparing the quality of one or more 
 * algorithms on a binary classification task. 
 * 
 * @author Edward Raff
 */
public class ROCPlot extends Graph2D
{
    /**
     * Index 0: the classifier 
     * Index 1: the x/y set of values
     * Index 2: the individual point values that belong to x or y
     */
    private double[][][] curves;
    
    private List<Color> categoryColors;
    private List<String> names;
    
    /**
     * Might have to compute the curves in the background before drawing
     */
    private volatile boolean readyToDraw;

    /**
     * Does some initialization. Sets up space for names, inits categoryColors, 
     * and intializes the first 2 dimensions of curves. 
     * @param dataSet the data set training/testing will be done from
     * @param numCurves the number of curves that are going to be ploted
     */
    private ROCPlot(int numCurves)
    {
        super(0.0, 1.0, 0.0, 1.0);
        
        readyToDraw = false;
        setYAxisTtile("True Positive Rate");
        setXAxisTtile("False Positive Rate");
        
        curves = new double[numCurves][2][];
        
        names = new ArrayList<String>(numCurves);
        categoryColors = GUIUtils.getDistinctColors(numCurves);
    }
    
    /**
     * Creates a new ROC plot from a given list of model evaluations. The models 
     * have to have been evaluated with 
     * {@link ClassificationModelEvaluation#keepPredictions(boolean) } as true. 
     * 
     * @param names the names to use for each of the model evaluations when 
     * plotting
     * @param evaluations the already computed model evaluations
     */
    public ROCPlot(List<String> names, ClassificationModelEvaluation... evaluations)
    {
        this(names.size());
        this.names.addAll(names);
        for (int i = 0; i < evaluations.length; i++)
        {
            curves[i][0] = new double[evaluations[i].getTruths().length];
            curves[i][1] = new double[curves[i][0].length];
        }
        
        computeCurves(Arrays.asList(evaluations));
    }
    
    /**
     * Creates a new ROC plot from a given set of classifiers and a data set to 
     * perform cross validation on. 
     * 
     * @param dataSet the data set to perform compute the ROC curve from with 
     * cross validation
     * @param folds the number of folds of cross validation to perform
     * @param classifiers the classifiers to test
     */
    public ROCPlot(final ClassificationDataSet dataSet, final int folds, final Classifier... classifiers)
    {
        this(classifiers.length);
        if(dataSet.getPredicting().getNumOfCategories() != 2)
            throw new RuntimeException("ROC curves can only be done for binarry classification problems");

        final List<ClassificationModelEvaluation> cmes =
                new ArrayList<ClassificationModelEvaluation>(classifiers.length);
        Thread thread = new Thread(new Runnable()
        {
            @Override
            public void run()
            {
                for (int i = 0; i < classifiers.length; i++)
                {
                    names.add(classifiers[i].getClass().getSimpleName());
                    curves[i][0] = new double[dataSet.getSampleSize()];
                    curves[i][1] = new double[dataSet.getSampleSize()];

                    ClassificationModelEvaluation cme = new ClassificationModelEvaluation(classifiers[i].clone(), dataSet);
                    cme.keepPredictions(true);
                    cme.evaluateCrossValidation(folds);
                    cmes.add(cme);
                }

                computeCurves(cmes);
            }
        });
        
        thread.start();
    }

    @Override
    protected void paintWork(Graphics g, int imageWidth, int imageHeight, ProgressPanel pp)
    {
        super.paintWork(g, imageWidth, imageHeight, pp);
        
        if(!readyToDraw)
        {
            String message = "Computing ROC Curves...";
            int strWidth = g.getFontMetrics().stringWidth(message);
            g.drawString(message, imageWidth/2-strWidth/2, imageHeight/2);
            return;
        }
        double prevTP;
        double prevFP;
        
        g.setColor(Color.DARK_GRAY);
        g.drawLine(toXCord(0.0), toYCord(0.0), toXCord(1.0), toYCord(1.0));
        
        for (int ci = 0; ci < curves.length; ci++)
        {
            g.setColor(categoryColors.get(ci));
            prevTP = 0.0;
            prevFP = 0.0;
            for (int i = 0; i < curves[ci][0].length - 1; i++)
            {
                
                double nextTP = curves[ci][0][i];
                double nextFP = curves[ci][1][i];
                
                g.drawLine(toXCord(prevFP), toYCord(prevTP), toXCord(nextFP), toYCord(nextTP));
                prevFP = nextFP;
                prevTP = nextTP;
            }
        }
        
        drawKey((Graphics2D)g, 3, names, categoryColors, null);
    }

    private void computeCurves(final List<ClassificationModelEvaluation> cmes)
    {
        for (int ci = 0; ci < cmes.size(); ci++)
        {
            ClassificationModelEvaluation cme = cmes.get(ci);

            CategoricalResults[] results = cme.getPredictions();

            IndexTable it = new IndexTable(Arrays.asList(results),
                    new Comparator<CategoricalResults>()
                    {
                        @Override
                        public int compare(CategoricalResults t, CategoricalResults t1)
                        {
                            return -Double.compare(t.getProb(0), t1.getProb(0));
                        }
                    });

            int[] truth = cme.getTruths();
            double[] weights = cme.getPointWeights();
            //TODO re-write this as one loop over i
            for (int i = 0; i < results.length; i++)
            {
                double TP = 0.0;
                double TN = 0.0;
                double FN = 0.0;
                double FP = 0.0;
                int origIndx = it.index(i);
                double thresh = results[origIndx].getProb(0);

                for (int j = 0; j < results.length; j++)
                {

                    double weight = weights[j];
                    int trueClass = truth[j];
                    int predClass = results[j].getProb(0) >= thresh ? 0 : 1;

                    if (trueClass == 0 && predClass == 0)
                        TP += weight;
                    else if (trueClass == 1 && predClass == 1)
                        TN += weight;
                    else if (trueClass == 1 && predClass == 0)
                        FP += weight;
                    else if (trueClass == 0 && predClass == 1)
                        FN += weight;
                }

                curves[ci][0][i] = TP / (TP + FN);
                if (FP + TN > 0)
                    curves[ci][1][i] = FP / (FP + TN);
                else
                    curves[ci][1][i] = 1.0;
            }
        }

        readyToDraw = true;
        forceRedraw();
        repaint();
    }

}
