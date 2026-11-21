#ifndef GUI_PREFERENCES_DIALOG_HPP
#define GUI_PREFERENCES_DIALOG_HPP

#include "app_settings.hpp"

#include <QDialog>

class QCheckBox;
class QComboBox;
class QLineEdit;
class QPushButton;
class QSpinBox;

class PreferencesDialog final : public QDialog {
    Q_OBJECT

  public:
    explicit PreferencesDialog(GuiSettings settings, QWidget* parent = nullptr);
    ~PreferencesDialog() override = default;
    PreferencesDialog(const PreferencesDialog&) = delete;
    PreferencesDialog& operator=(const PreferencesDialog&) = delete;
    PreferencesDialog(PreferencesDialog&&) = delete;
    PreferencesDialog& operator=(PreferencesDialog&&) = delete;

    [[nodiscard]] GuiSettings settings() const { return settings_; }

  protected:
    void accept() override;

  private:
    void browse_engine();
    void browse_eval_file();
    [[nodiscard]] GuiSettings settings_from_widgets() const;

    GuiSettings settings_;
    QLineEdit* engine_path_edit_{nullptr};
    QSpinBox* hash_spin_{nullptr};
    QSpinBox* threads_spin_{nullptr};
    QSpinBox* multi_pv_spin_{nullptr};
    QSpinBox* move_overhead_spin_{nullptr};
    QCheckBox* use_nnue_check_{nullptr};
    QLineEdit* eval_file_edit_{nullptr};
    QComboBox* nnue_backend_combo_{nullptr};
    QSpinBox* engine_move_time_spin_{nullptr};
};

#endif // GUI_PREFERENCES_DIALOG_HPP
