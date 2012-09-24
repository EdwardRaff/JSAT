
package jsat.guitool;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Frame;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import javax.swing.JDialog;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.ProgressMonitor;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.Classifier;
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
        
        if(dTime > 1000)
        {
            dTime/=1000;
            unit = " s";
        }
        if(dTime > 60*2)
        {
            dTime /= 60;
            unit = " m";
        }
        if(dTime > 60*2)
        {
            dTime /= 60;
            unit = " h";
        }
        
        return df.format(dTime) + unit;
    }
    
    public ClassifierCVEvaluation(final List<Classifier> classifiers, final List<String> classifierNames, final ClassificationDataSet dataset, Frame owner, String title, boolean modal)
    {
        super(owner, title, modal);
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
                for(int i = 0; i < classifiers.size() && !pm.isCanceled(); i++)
                {
                    final int ip1 = i+1;
                    try
                    {
                        Classifier classifier = classifiers.get(i);
                        String name = classifierNames.get(i);
                        CategoricalData predicting = dataset.getPredicting();

                        pm.setNote("Evaluating " + name);

                        final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(classifier, dataset, threadPool);
                        cme.evaluateCrossValidation(10);
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

                        panel.add(new JScrollPane(jList), BorderLayout.EAST);

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
                        if(pm.isCanceled())
                            return null;
                        else//something went really wrong!
                            resultPanels.add(new JPanel());
                    }
                    
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
                for(int i = 0; i < classifiers.size(); i++)
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
