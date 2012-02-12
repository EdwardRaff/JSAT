
package jsat.guitool;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Frame;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;

/**
 * This class is used to provide a dialog indicating progress for a long running task.
 * @author Edward Raff
 */
public class LongProcessDialog extends JDialog
{
    private JLabel messageLabel = new JLabel();
    private JLabel noteLabel = new JLabel();
    private JProgressBar progressBar;
    private JButton cancelButton = new JButton("Cancel");
    private volatile boolean canceled = false;
    private List<ActionListener> actions = new ArrayList<ActionListener>();
    
    public LongProcessDialog(Frame owner, String title)
    {
        super(owner, title, false);
        progressBar = new JProgressBar();
        progressBar.setIndeterminate(true);
        setLayout(new BorderLayout());
        JPanel tmp = new JPanel(new FlowLayout());
        tmp.add(cancelButton);
        add(tmp, BorderLayout.SOUTH);
        JPanel mainPanel = new JPanel(new GridLayout(3, 1));
        Font font = messageLabel.getFont();
        Font newFont = new Font(font.getName(), font.getStyle(), font.getSize()+2);
        messageLabel.setFont(newFont);
        tmp = new JPanel(new FlowLayout());
        tmp.add(messageLabel);
        mainPanel.add(tmp);
        
        tmp = new JPanel(new FlowLayout());
        tmp.add(noteLabel);
        mainPanel.add(tmp);
        
        tmp = new JPanel(new FlowLayout());
        tmp.add(progressBar);
        mainPanel.add(tmp);
        add(mainPanel, BorderLayout.CENTER);
        
        cancelButton.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e)
            {
                canceled = true;
                setVisible(false);
                for(ActionListener al: actions)
                    al.actionPerformed(e);
            }
        });
    }
    
    public void addCancleActionListener(ActionListener e)
    {
        actions.add(e);
    }

    public void setIndeterminate(boolean indeterminate)
    {
        progressBar.setIndeterminate(indeterminate);
    }
    
    public void setMessage(String message)
    {
        messageLabel.setText(message);
    }
    
    public void setNote(String note)
    {
        noteLabel.setText(note);
    }
    
    public void setMinimum(int i)
    {
        progressBar.setIndeterminate(false);
        progressBar.setMinimum(i);
    }
    
    public void setMaximum(int i)
    {
        progressBar.setIndeterminate(false);
        progressBar.setMaximum(i);
    }
    
    public void setValue(int i)
    {
        progressBar.setValue(i);
        if(progressBar.getValue() == progressBar.getMaximum())
            setVisible(false);
    }

    public boolean isCanceled()
    {
        return canceled;
    }
}
