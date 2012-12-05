
package jsat.guitool;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Frame;
import java.awt.GridLayout;
import java.awt.LayoutManager;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import javax.swing.*;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.Classifier;
import jsat.datatransform.DataTransformProcess;
import jsat.exceptions.FailedToFitException;
import jsat.graphing.ROCPlot;
import jsat.parameters.Parameterized;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class ClassifierCVEvaluation extends JDialog
{
    private ExecutorService threadPool;

    private static String timeString(long time)
    {
        DecimalFormat df = new DecimalFormat("0.####");
        String unit = " ms";
        
        double dTime = time;
        
        if (dTime > 1000)
        {
            dTime /= 1000;
            unit = " s";
            if (dTime > 60 * 2)
            {
                dTime /= 60;
                unit = " m";
                if (dTime > 60 * 2)
                {
                    dTime /= 60;
                    unit = " h";
                }
            }

        }
        
        return df.format(dTime) + unit;
    }
    
    public ClassifierCVEvaluation(final List<Classifier> classifiers, final List<String> classifierNames, final ClassificationDataSet dataset, Frame owner, String title, boolean modal, final DataTransformProcess dtp)
    {
        super(owner, title, modal);
        final boolean isBinaryProblem = dataset.getPredicting().getNumOfCategories() == 2;
        final LongProcessDialog pm = new LongProcessDialog(owner, "Performing Cross Validation");
        pm.setMinimum(0);
        pm.setMaximum(classifiers.size());
        pm.setMessage("Performing Cross Validation");
        pm.setNote("Note");
        pm.pack();
        pm.setSize(450, pm.getHeight());
        pm.setVisible(true);
        
        final List<Thread> threadsCreated = new ArrayList<Thread>();
        ThreadFactory threadFactory = new ThreadFactory() {

            @Override
            public Thread newThread(Runnable r)
            {
                Thread toReturn = new Thread(r);
                toReturn.setDaemon(false);
                threadsCreated.add(toReturn);
                return toReturn;
            }
        };
        this.threadPool = Executors.newFixedThreadPool(SystemInfo.LogicalCores, threadFactory);
        
        
        final List<JPanel> resultPanels = new ArrayList<JPanel>();

        final SwingWorker worker = new SwingWorker() 
        {
            @Override
            protected Object doInBackground() throws Exception
            {
                DecimalFormat df = new DecimalFormat("0.00000");
                List<ClassificationModelEvaluation> cmes = new ArrayList<ClassificationModelEvaluation>();
                for(int i = 0; i < classifiers.size() && !pm.isCanceled(); i++)
                {
                    final int ip1 = i+1;
                    try
                    {
                        Classifier classifier = classifiers.get(i);
                        String name = classifierNames.get(i);
                        CategoricalData predicting = dataset.getPredicting();

                        pm.setNote("Evaluating " + name);

                        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(classifier, dataset, threadPool);
                        if(isBinaryProblem)
                        {
                            cme.keepPredictions(true);
                            cmes.add(cme);
                        }
                        if (dtp != null)
                            cme.setDataTransformProcess(dtp);
                        try
                        {
                            cme.evaluateCrossValidation(10);
                        }
                        catch (FailedToFitException ex)
                        {
                            if (pm.isCanceled())
                                return null;
                            pm.setNote("Evaluating " + name + ", falling back to Single Threaded");
                            try
                            {
                                cme = new ClassificationModelEvaluation(classifier, dataset);
                                cme.evaluateCrossValidation(10);
                            }
                            catch (Exception exexp)
                            {
                                
                            }
                        }
                        if (pm.isCanceled())
                            return null;
                        JPanel panel = new JPanel(new BorderLayout());
                        double[][] confusionMatrix = cme.getConfusionMatrix();
                        String[] columnNames = new String[predicting.getNumOfCategories() + 1];
                        Object[][] data = new Object[columnNames.length - 1][columnNames.length];

                        columnNames[0] = "Confusion Matrix";
                        for (int j = 0; j < data.length; j++)
                        {
                            data[j][0] = columnNames[j + 1] = predicting.getOptionName(j);
                            for (int k = 1; k < data[j].length; k++)
                                data[j][k] = new Double(confusionMatrix[j][k - 1]);
                        }

                        panel.add(new JScrollPane(new JTable(data, columnNames)), BorderLayout.CENTER);

                        Vector listInfo = new Vector();
                        listInfo.add("Error Rate: " + df.format(cme.getErrorRate()));
                        listInfo.add("Training Time: " + timeString(cme.getTotalTrainingTime()));
                        listInfo.add("Classification Time: " + timeString(cme.getTotalClassificationTime()));
                        JList jList = new JList(listInfo);
                        
                        LayoutManager layout;
                        
                        if(classifier instanceof Parameterized)
                            layout = new GridLayout(2, 1);
                        else
                            layout = new GridLayout(1, 1);
                        
                        JPanel rightPanel = new JPanel(layout);
                        rightPanel.add(new JScrollPane(jList));
                        
                        if(classifier instanceof Parameterized)
                        {
                            JPanel sub = new JPanel();
                            ParameterPanel pp = new ParameterPanel((Parameterized)classifier);
                            pp.getjButtonOk().setVisible(false);
                            sub.add(pp);
                            //Disable all fields, we dont want them making changes
                            Stack<JComponent> components = new Stack<JComponent>();
                            components.add(pp);
                            while(!components.isEmpty())
                            {
                                for(Component comp : components.pop().getComponents())
                                {
                                    comp.setEnabled(false);
                                    if(comp instanceof JComponent)
                                        components.add((JComponent)comp);
                                }
                            }
                            rightPanel.add(sub);
                        }

                        panel.add(rightPanel, BorderLayout.EAST);

                        resultPanels.add(panel);

                        SwingUtilities.invokeLater(new Runnable()
                        {

                            public void run()
                            {
                                pm.setValue(ip1);
                            }
                        });
                    }
                    catch(Exception ex)
                    {
                        //TODO set error box
                        if(pm.isCanceled())
                            return null;
                        else//something went really wrong!
                        {
                            JPanel jp = new JPanel(new GridLayout(1, 1));
                            JTextArea jta = new JTextArea("The folloing error occured:\n" + ex.getMessage());
                            jp.add(jta);
                            resultPanels.add(jp);
                        }
                    }
                    
                }
                
                if(isBinaryProblem)
                {
                    ROCPlot rocPlot = new ROCPlot(new ArrayList<String>(classifierNames), cmes);
                    classifierNames.add("ROC Curves");
                    JPanel jp = new JPanel(new GridLayout(1, 1));
                    jp.add(rocPlot);
                    resultPanels.add(jp);
                }
                
                return null;
            }
            
            @Override
            protected void done()
            {
                if(pm.isCanceled())
                {    
                    setVisible(false);
                    threadPool.shutdownNow();
                    return;
                }
                JTabbedPane jTabbedPane = new JTabbedPane();
                for(int i = 0; i < classifierNames.size(); i++)
                {
                    jTabbedPane.add(classifierNames.get(i), resultPanels.get(i));
                }
                
                setLayout(new GridLayout(1, 1));
                add(jTabbedPane);
                pack();
                setVisible(true);
                threadPool.shutdown();
            }
        };
        
        pm.addCancleActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e)
            {
                worker.cancel(true);
                threadPool.shutdownNow();
                for (Thread t : threadsCreated)
                    if (t.isAlive())
                        try
                        {
                            //TODO, how do i get this thing to stop printing errors?
                            t.stop();//Depricated, but onlny way to force stop
                        }
                        catch (Exception ex)
                        {
                        }

            }
        });
        
        worker.execute();
    }
    
    
}
