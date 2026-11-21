#include "preferences_dialog.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QStyle>
#include <QVBoxLayout>

#include <utility>

namespace {

constexpr int kDialogMinimumWidth = 520;

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] QSpinBox* spin_box(int minimum, int maximum, int value, QWidget* parent,
                                 const QString& suffix = {}) {
    auto* spin = new QSpinBox(parent); // NOLINT(cppcoreguidelines-owning-memory)
    spin->setRange(minimum, maximum);
    spin->setValue(value);
    spin->setAccelerated(true);
    if (!suffix.isEmpty()) {
        spin->setSuffix(suffix);
    }
    return spin;
}

[[nodiscard]] QWidget* path_row(QLineEdit* edit, QPushButton* button, QWidget* parent) {
    auto* row = new QWidget(parent);     // NOLINT(cppcoreguidelines-owning-memory)
    auto* layout = new QHBoxLayout(row); // NOLINT(cppcoreguidelines-owning-memory)
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(edit, 1);
    layout->addWidget(button);
    return row;
}

} // namespace

// NOLINTBEGIN(cppcoreguidelines-owning-memory, cppcoreguidelines-prefer-member-initializer)
PreferencesDialog::PreferencesDialog(GuiSettings settings, QWidget* parent)
    : QDialog(parent), settings_{std::move(settings)} {
    setWindowTitle(QStringLiteral("Preferences"));
    setModal(true);
    setMinimumWidth(kDialogMinimumWidth);

    auto* root = new QVBoxLayout(this);

    auto* engine_group = new QGroupBox(QStringLiteral("Engine"), this);
    auto* engine_form = new QFormLayout(engine_group);
    engine_path_edit_ = new QLineEdit(settings_.engine_path, engine_group);
    auto* engine_browse = new QPushButton(style()->standardIcon(QStyle::SP_DialogOpenButton),
                                          QStringLiteral("Browse..."), engine_group);
    connect(engine_browse, &QPushButton::clicked, this, &PreferencesDialog::browse_engine);
    engine_form->addRow(QStringLiteral("Executable"),
                        path_row(engine_path_edit_, engine_browse, engine_group));
    hash_spin_ = spin_box(GuiSettings::min_hash_mb, GuiSettings::max_hash_mb, settings_.hash_mb,
                          engine_group, QStringLiteral(" MB"));
    threads_spin_ = spin_box(GuiSettings::min_threads, GuiSettings::max_threads, settings_.threads,
                             engine_group);
    multi_pv_spin_ = spin_box(GuiSettings::min_multi_pv, GuiSettings::max_multi_pv,
                              settings_.multi_pv, engine_group);
    move_overhead_spin_ =
        spin_box(GuiSettings::min_move_overhead_ms, GuiSettings::max_move_overhead_ms,
                 settings_.move_overhead_ms, engine_group, QStringLiteral(" ms"));
    engine_form->addRow(QStringLiteral("Hash"), hash_spin_);
    engine_form->addRow(QStringLiteral("Threads"), threads_spin_);
    engine_form->addRow(QStringLiteral("MultiPV"), multi_pv_spin_);
    engine_form->addRow(QStringLiteral("Move Overhead"), move_overhead_spin_);
    root->addWidget(engine_group);

    auto* search_group = new QGroupBox(QStringLiteral("Search"), this);
    auto* search_form = new QFormLayout(search_group);
    engine_move_time_spin_ =
        spin_box(GuiSettings::min_engine_move_time_ms, GuiSettings::max_engine_move_time_ms,
                 settings_.engine_move_time_ms, search_group, QStringLiteral(" ms"));
    search_form->addRow(QStringLiteral("Engine move time"), engine_move_time_spin_);
    root->addWidget(search_group);

    auto* nnue_group = new QGroupBox(QStringLiteral("NNUE"), this);
    auto* nnue_form = new QFormLayout(nnue_group);
    use_nnue_check_ = new QCheckBox(QStringLiteral("Use NNUE"), nnue_group);
    use_nnue_check_->setChecked(settings_.use_nnue);
    eval_file_edit_ = new QLineEdit(settings_.eval_file, nnue_group);
    auto* eval_browse = new QPushButton(style()->standardIcon(QStyle::SP_DialogOpenButton),
                                        QStringLiteral("Browse..."), nnue_group);
    connect(eval_browse, &QPushButton::clicked, this, &PreferencesDialog::browse_eval_file);
    nnue_backend_combo_ = new QComboBox(nnue_group);
    for (const auto backend : {GuiSettings::NnueBackend::Scalar, GuiSettings::NnueBackend::Neon,
                               GuiSettings::NnueBackend::Auto}) {
        nnue_backend_combo_->addItem(GuiSettings::nnue_backend_to_string(backend),
                                     static_cast<int>(backend));
    }
    nnue_backend_combo_->setCurrentIndex(
        nnue_backend_combo_->findData(static_cast<int>(settings_.nnue_backend)));
    nnue_form->addRow({}, use_nnue_check_);
    nnue_form->addRow(QStringLiteral("EvalFile"),
                      path_row(eval_file_edit_, eval_browse, nnue_group));
    nnue_form->addRow(QStringLiteral("Backend"), nnue_backend_combo_);
    root->addWidget(nnue_group);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(buttons, &QDialogButtonBox::accepted, this, &PreferencesDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &PreferencesDialog::reject);
    root->addWidget(buttons);
}
// NOLINTEND(cppcoreguidelines-owning-memory, cppcoreguidelines-prefer-member-initializer)

void PreferencesDialog::accept() {
    const GuiSettings proposed = settings_from_widgets();
    const QString error = proposed.validation_error();
    if (!error.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("Preferences"), error);
        return;
    }
    settings_ = proposed;
    QDialog::accept();
}

void PreferencesDialog::browse_engine() {
    const QString start_dir = engine_path_edit_->text().isEmpty()
                                  ? QString{}
                                  : QFileInfo{engine_path_edit_->text().trimmed()}.absolutePath();
    const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Select Engine"),
                                                      start_dir, QStringLiteral("Executables (*)"));
    if (!path.isEmpty()) {
        engine_path_edit_->setText(path);
    }
}

void PreferencesDialog::browse_eval_file() {
    const QString start_dir = eval_file_edit_->text().isEmpty()
                                  ? QString{}
                                  : QFileInfo{eval_file_edit_->text().trimmed()}.absolutePath();
    const QString path =
        QFileDialog::getOpenFileName(this, QStringLiteral("Select NNUE File"), start_dir,
                                     QStringLiteral("NNUE Files (*.nnue);;All Files (*)"));
    if (!path.isEmpty()) {
        eval_file_edit_->setText(path);
    }
}

GuiSettings PreferencesDialog::settings_from_widgets() const {
    GuiSettings proposed = settings_;
    proposed.engine_path = engine_path_edit_->text().trimmed();
    proposed.hash_mb = hash_spin_->value();
    proposed.threads = threads_spin_->value();
    proposed.multi_pv = multi_pv_spin_->value();
    proposed.move_overhead_ms = move_overhead_spin_->value();
    proposed.use_nnue = use_nnue_check_->isChecked();
    proposed.eval_file = eval_file_edit_->text().trimmed();
    proposed.nnue_backend =
        static_cast<GuiSettings::NnueBackend>(nnue_backend_combo_->currentData().toInt());
    proposed.engine_move_time_ms = engine_move_time_spin_->value();
    return proposed;
}
