#include "main_window.hpp"

#include "app_icon.hpp"
#include "board_widget.hpp"
#include "engine_controller.hpp"
#include "evaluation_bar_widget.hpp"
#include "evaluation_display.hpp"
#include "game_sidebar.hpp"
#include "pgn.hpp"
#include "preferences_dialog.hpp"

#include <eng/movegen.hpp>

#include <QAction>
#include <QApplication>
#include <QByteArray>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QIcon>
#include <QImage>
#include <QImageWriter>
#include <QKeySequence>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QSaveFile>
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QStringList>
#include <QStyle>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kWindowMinWidth = 640;
constexpr int kWindowMinHeight = 480;
constexpr int kWindowStartWidth = 1040;
constexpr int kWindowStartHeight = 720;
constexpr int kContentMargin = 16;
constexpr int kBundleBuildDirUpLevels = 4;
constexpr eng::Color kEngineColor = eng::Color::Black;

[[nodiscard]] QString color_name(eng::Color color) {
    return color == eng::Color::Black ? QStringLiteral("Black") : QStringLiteral("White");
}

[[nodiscard]] QString promotion_name(eng::PieceType piece) {
    switch (piece) {
    case eng::PieceType::Queen:
        return QStringLiteral("Queen");
    case eng::PieceType::Rook:
        return QStringLiteral("Rook");
    case eng::PieceType::Bishop:
        return QStringLiteral("Bishop");
    case eng::PieceType::Knight:
        return QStringLiteral("Knight");
    default:
        return QStringLiteral("Piece");
    }
}

[[nodiscard]] QChar promotion_char(eng::PieceType piece) {
    switch (piece) {
    case eng::PieceType::Queen:
        return QLatin1Char('q');
    case eng::PieceType::Rook:
        return QLatin1Char('r');
    case eng::PieceType::Bishop:
        return QLatin1Char('b');
    case eng::PieceType::Knight:
        return QLatin1Char('n');
    case eng::PieceType::Pawn:
    case eng::PieceType::King:
    case eng::PieceType::None:
        break;
    }
    return {};
}

[[nodiscard]] QString square_to_uci(eng::Square square) {
    QString result;
    result.reserve(2);
    result.append(QChar{static_cast<char16_t>('a' + square.file().index())});
    result.append(QChar{static_cast<char16_t>('1' + square.rank().index())});
    return result;
}

[[nodiscard]] QString move_to_uci(eng::Move move) {
    if (!move.is_valid() || move.is_null()) {
        return QStringLiteral("0000");
    }
    QString result = square_to_uci(move.from()) + square_to_uci(move.to());
    if (move.is_promotion()) {
        result.append(promotion_char(move.promotion_piece()));
    }
    return result;
}

[[nodiscard]] std::optional<eng::Move> legal_move_from_uci(const eng::Position& position,
                                                           const QString& notation) {
    const QString normalized = notation.trimmed().toLower();
    const auto legal_moves = eng::generate_legal_moves(position);
    for (const auto move : legal_moves) {
        if (move_to_uci(move) == normalized) {
            return move;
        }
    }
    return std::nullopt;
}

[[nodiscard]] QString pgn_error_text(const gui::pgn::Error& error) {
    return QStringLiteral("%1 (offset %2)")
        .arg(QString::fromStdString(error.message))
        .arg(static_cast<qulonglong>(error.offset));
}

[[nodiscard]] QString pgn_warning_text(const gui::pgn::Game& game) {
    if (game.warnings.empty()) {
        return {};
    }
    return game.warnings.size() == 1U
               ? QStringLiteral("Loaded PGN with warning: %1")
                     .arg(QString::fromStdString(game.warnings.front().message))
               : QStringLiteral("Loaded PGN with %1 warnings.")
                     .arg(static_cast<qulonglong>(game.warnings.size()));
}

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
void append_be32(QByteArray& bytes, std::uint32_t value) {
    bytes.append(static_cast<char>((value >> 24U) & 0xFFU));
    bytes.append(static_cast<char>((value >> 16U) & 0xFFU));
    bytes.append(static_cast<char>((value >> 8U) & 0xFFU));
    bytes.append(static_cast<char>(value & 0xFFU));
}

void append_le16(QByteArray& bytes, std::uint16_t value) {
    bytes.append(static_cast<char>(value & 0xFFU));
    bytes.append(static_cast<char>((value >> 8U) & 0xFFU));
}

[[nodiscard]] std::uint32_t crc32(const QByteArray& bytes) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const auto byte : bytes) {
        crc ^= static_cast<unsigned char>(byte);
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0U - (crc & 1U);
            crc = (crc >> 1U) ^ (0xEDB88320U & mask);
        }
    }
    return crc ^ 0xFFFFFFFFU;
}

[[nodiscard]] std::uint32_t adler32(const QByteArray& bytes) {
    std::uint32_t low = 1U;
    std::uint32_t high = 0U;
    for (const auto byte : bytes) {
        low = (low + static_cast<unsigned char>(byte)) % 65521U;
        high = (high + low) % 65521U;
    }
    return (high << 16U) | low;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void append_png_chunk(QByteArray& png, const char* chunk_type, const QByteArray& payload) {
    append_be32(png, static_cast<std::uint32_t>(payload.size()));
    png.append(chunk_type, 4);
    png.append(payload);

    QByteArray crc_input;
    crc_input.reserve(4 + payload.size());
    crc_input.append(chunk_type, 4);
    crc_input.append(payload);
    append_be32(png, crc32(crc_input));
}

[[nodiscard]] QByteArray stored_zlib_stream(const QByteArray& raw) {
    constexpr int kMaxStoredBlockSize = 65535;

    QByteArray stream;
    stream.reserve(raw.size() + (raw.size() / kMaxStoredBlockSize * 5) + 8);
    stream.append(static_cast<char>(0x78));
    stream.append(static_cast<char>(0x01));

    if (raw.isEmpty()) {
        stream.append(static_cast<char>(0x01));
        append_le16(stream, std::uint16_t{0});
        append_le16(stream, std::uint16_t{0xFFFFU});
    }

    for (qsizetype offset = 0; offset < raw.size();) {
        const qsizetype remaining = raw.size() - offset;
        const int block_size =
            static_cast<int>(std::min<qsizetype>(remaining, kMaxStoredBlockSize));
        const bool final_block = offset + block_size >= raw.size();
        const auto stored_size = static_cast<std::uint16_t>(block_size);
        stream.append(static_cast<char>(final_block ? 0x01 : 0x00));
        append_le16(stream, stored_size);
        append_le16(stream, static_cast<std::uint16_t>(0xFFFFU - stored_size));
        stream.append(raw.constData() + offset, block_size);
        offset += block_size;
    }

    append_be32(stream, adler32(raw));
    return stream;
}

[[nodiscard]] bool save_png_fallback(const QImage& image, const QString& path) {
    if (image.isNull()) {
        return false;
    }

    const QImage rgba = image.convertToFormat(QImage::Format_RGBA8888);
    const qsizetype bytes_per_rgba_row = static_cast<qsizetype>(rgba.width()) * 4;
    QByteArray raw;
    raw.reserve((bytes_per_rgba_row + 1) * static_cast<qsizetype>(rgba.height()));
    for (int row = 0; row < rgba.height(); ++row) {
        raw.append(static_cast<char>(0x00));
        raw.append(reinterpret_cast<const char*>(rgba.constScanLine(row)), bytes_per_rgba_row);
    }

    QByteArray png;
    png.append("\x89PNG\r\n\x1A\n", 8);

    QByteArray ihdr;
    append_be32(ihdr, static_cast<std::uint32_t>(rgba.width()));
    append_be32(ihdr, static_cast<std::uint32_t>(rgba.height()));
    ihdr.append(static_cast<char>(8));
    ihdr.append(static_cast<char>(6));
    ihdr.append(static_cast<char>(0));
    ihdr.append(static_cast<char>(0));
    ihdr.append(static_cast<char>(0));
    append_png_chunk(png, "IHDR", ihdr);
    append_png_chunk(png, "IDAT", stored_zlib_stream(raw));
    append_png_chunk(png, "IEND", {});

    QFile file{path};
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        return false;
    }
    return file.write(png) == png.size();
}

[[nodiscard]] bool save_widget_snapshot(QWidget* widget, const QString& path) {
    if (widget == nullptr || path.isEmpty()) {
        return false;
    }

    const QFileInfo output_info{path};
    const QString output_dir = output_info.absolutePath();
    if (!output_dir.isEmpty() && !QDir{}.mkpath(output_dir)) {
        return false;
    }

    QApplication::processEvents();
    QSize logical_size = widget->size();
    if (logical_size.isEmpty()) {
        logical_size = widget->sizeHint();
        widget->resize(logical_size);
    }

    const qreal device_pixel_ratio = widget->devicePixelRatioF();
    const QSize image_size{
        std::max(1, static_cast<int>(std::ceil(logical_size.width() * device_pixel_ratio))),
        std::max(1, static_cast<int>(std::ceil(logical_size.height() * device_pixel_ratio)))};
    QImage snapshot{image_size, QImage::Format_ARGB32_Premultiplied};
    snapshot.setDevicePixelRatio(device_pixel_ratio);
    snapshot.fill(Qt::transparent);

    QPainter painter{&snapshot};
    widget->render(&painter);
    painter.end();

    const bool qt_can_write_png = QImageWriter::supportedImageFormats().contains("png");
    if (qt_can_write_png && snapshot.save(path, "PNG")) {
        return true;
    }
    return save_png_fallback(snapshot, path);
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)

} // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("LocalChessEngine"));
    setWindowIcon(make_app_icon());
    setMinimumSize(kWindowMinWidth, kWindowMinHeight);
    resize(kWindowStartWidth, kWindowStartHeight);
    setUnifiedTitleAndToolBarOnMac(true);

    settings_ = GuiSettings::load(default_engine_path());
    create_engine_controller();
    create_menus();
    create_central_area();
    install_status_bar_branding();
    sync_board_from_game();
}

MainWindow::~MainWindow() {
    if (engine_controller_ == nullptr) {
        return;
    }
    QObject::disconnect(engine_controller_, nullptr, this, nullptr);
    if (engine_controller_->is_running()) {
        engine_controller_->quit_engine();
    }
}

bool MainWindow::save_board_snapshot(const QString& path) const {
    return save_widget_snapshot(board_widget_, path);
}

bool MainWindow::save_window_snapshot(const QString& path) {
    return save_widget_snapshot(this, path);
}

void MainWindow::create_engine_controller() {
    engine_controller_ = new EngineController(this); // NOLINT(cppcoreguidelines-owning-memory)
    apply_settings_to_engine();

    connect(engine_controller_, &EngineController::stateChanged, this,
            [this](EngineController::State) { handle_engine_state_changed(); });
    connect(engine_controller_, &EngineController::searchInfoReceived, this,
            [this](const EngineController::SearchInfo& info) { handle_engine_search_info(info); });
    connect(engine_controller_, &EngineController::bestMoveReceived, this,
            [this](const EngineController::BestMove& bestmove) {
                handle_engine_bestmove(bestmove.move);
            });
    connect(engine_controller_, &EngineController::errorOccurred, this,
            [this](const QString& message) {
                awaiting_engine_move_ = false;
                pending_engine_turn_ = false;
                pending_analysis_ = false;
                restart_engine_when_stopped_ = false;
                active_search_purpose_ = SearchPurpose::None;
                active_search_side_to_move_.reset();
                sync_board_from_game(QStringLiteral("Engine error: %1").arg(message));
            });
    connect(engine_controller_, &EngineController::engineEvent, this,
            [this](const QString& event_type, const QString& detail) {
                handle_engine_event(event_type, detail);
            });
}

void MainWindow::create_menus() {
    menuBar()->setNativeMenuBar(true);

    auto* file_menu = menuBar()->addMenu(QStringLiteral("&File"));
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    open_pgn_action_ = new QAction(style()->standardIcon(QStyle::SP_DialogOpenButton),
                                   QStringLiteral("&Open PGN..."), this);
    open_pgn_action_->setShortcut(QKeySequence::Open);
    connect(open_pgn_action_, &QAction::triggered, this, [this]() { open_pgn(); });
    file_menu->addAction(open_pgn_action_);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    save_pgn_action_ = new QAction(style()->standardIcon(QStyle::SP_DialogSaveButton),
                                   QStringLiteral("&Save PGN"), this);
    save_pgn_action_->setShortcut(QKeySequence::Save);
    connect(save_pgn_action_, &QAction::triggered, this, [this]() { save_pgn(); });
    file_menu->addAction(save_pgn_action_);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    save_pgn_as_action_ = new QAction(QStringLiteral("Save PGN &As..."), this);
    save_pgn_as_action_->setShortcut(QKeySequence::SaveAs);
    connect(save_pgn_as_action_, &QAction::triggered, this, [this]() { save_pgn_as(); });
    file_menu->addAction(save_pgn_as_action_);
    file_menu->addSeparator();

    auto* new_game_action =
        new QAction(style()->standardIcon(QStyle::SP_FileIcon), QStringLiteral("&New Game"), this);
    new_game_action->setShortcut(QKeySequence::New);
    connect(new_game_action, &QAction::triggered, this, [this]() { new_game(); });
    file_menu->addAction(new_game_action);
    file_menu->addSeparator();

    auto* quit_action = new QAction(style()->standardIcon(QStyle::SP_DialogCloseButton),
                                    QStringLiteral("&Quit"), this);
    quit_action->setShortcut(QKeySequence::Quit);
    connect(quit_action, &QAction::triggered, this, &QWidget::close);
    file_menu->addAction(quit_action);

    auto* game_menu = menuBar()->addMenu(QStringLiteral("&Game"));
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    first_move_action_ = new QAction(style()->standardIcon(QStyle::SP_MediaSkipBackward),
                                     QStringLiteral("First Move"), this);
    connect(first_move_action_, &QAction::triggered, this, [this]() { navigate_first(); });
    game_menu->addAction(first_move_action_);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    previous_move_action_ = new QAction(style()->standardIcon(QStyle::SP_MediaSeekBackward),
                                        QStringLiteral("Previous Move"), this);
    connect(previous_move_action_, &QAction::triggered, this, [this]() { navigate_previous(); });
    game_menu->addAction(previous_move_action_);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    next_move_action_ = new QAction(style()->standardIcon(QStyle::SP_MediaSeekForward),
                                    QStringLiteral("Next Move"), this);
    connect(next_move_action_, &QAction::triggered, this, [this]() { navigate_next(); });
    game_menu->addAction(next_move_action_);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    last_move_action_ = new QAction(style()->standardIcon(QStyle::SP_MediaSkipForward),
                                    QStringLiteral("Last Move"), this);
    connect(last_move_action_, &QAction::triggered, this, [this]() { navigate_last(); });
    game_menu->addAction(last_move_action_);
    update_pgn_actions();

    auto* engine_menu = menuBar()->addMenu(QStringLiteral("&Engine"));
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    start_engine_action_ = new QAction(style()->standardIcon(QStyle::SP_MediaPlay),
                                       QStringLiteral("&Start Engine"), this);
    connect(start_engine_action_, &QAction::triggered, this, [this]() { start_engine(); });
    engine_menu->addAction(start_engine_action_);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    analysis_toggle_action_ = new QAction(QStringLiteral("&Pause Analysis"), this);
    connect(analysis_toggle_action_, &QAction::triggered, this,
            [this]() { toggle_analysis_pause(); });
    engine_menu->addAction(analysis_toggle_action_);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    stop_search_action_ = new QAction(style()->standardIcon(QStyle::SP_MediaStop),
                                      QStringLiteral("Stop &Search"), this);
    connect(stop_search_action_, &QAction::triggered, this, [this]() { stop_engine_search(); });
    engine_menu->addAction(stop_search_action_);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    stop_engine_action_ = new QAction(style()->standardIcon(QStyle::SP_DialogCloseButton),
                                      QStringLiteral("Stop &Engine"), this);
    connect(stop_engine_action_, &QAction::triggered, this, [this]() { stop_engine(); });
    engine_menu->addAction(stop_engine_action_);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    restart_engine_action_ = new QAction(style()->standardIcon(QStyle::SP_BrowserReload),
                                         QStringLiteral("&Restart Engine"), this);
    connect(restart_engine_action_, &QAction::triggered, this, [this]() { restart_engine(); });
    engine_menu->addAction(restart_engine_action_);

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    preferences_action_ = new QAction(style()->standardIcon(QStyle::SP_FileDialogDetailedView),
                                      QStringLiteral("&Preferences..."), this);
    preferences_action_->setMenuRole(QAction::PreferencesRole);
    connect(preferences_action_, &QAction::triggered, this, [this]() { open_preferences(); });
    engine_menu->addAction(preferences_action_);
    engine_menu->addSeparator();

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    play_engine_action_ = new QAction(QStringLiteral("Play vs Engine (Human White)"), this);
    play_engine_action_->setCheckable(true);
    connect(play_engine_action_, &QAction::triggered, this, [this](bool enabled) {
        set_play_mode(enabled ? GameSidebar::PlayMode::HumanVsEngine
                              : GameSidebar::PlayMode::HumanVsHuman);
    });
    engine_menu->addAction(play_engine_action_);
    update_engine_actions();

    auto* help_menu = menuBar()->addMenu(QStringLiteral("&Help"));
    auto* about_action = new QAction(style()->standardIcon(QStyle::SP_MessageBoxInformation),
                                     QStringLiteral("&About"), this);
    connect(about_action, &QAction::triggered, this, [this]() { show_about_dialog(); });
    help_menu->addAction(about_action);

    auto* license_action = new QAction(style()->standardIcon(QStyle::SP_FileDialogDetailedView),
                                       QStringLiteral("&Third-Party Notices"), this);
    connect(license_action, &QAction::triggered, this, [this]() { show_license_dialog(); });
    help_menu->addAction(license_action);
}

void MainWindow::create_central_area() {
    auto* root = new QWidget(this); // NOLINT(cppcoreguidelines-owning-memory)
    root->setObjectName(QStringLiteral("mainContent"));
    root->setAccessibleName(QStringLiteral("Main content"));

    auto* layout = new QVBoxLayout(root); // NOLINT(cppcoreguidelines-owning-memory)
    layout->setContentsMargins(kContentMargin, kContentMargin, kContentMargin, kContentMargin);
    layout->setSpacing(0);

    auto* splitter = new QSplitter(Qt::Horizontal, root); // NOLINT(cppcoreguidelines-owning-memory)
    splitter->setObjectName(QStringLiteral("mainSplitter"));
    splitter->setChildrenCollapsible(false);

    auto* board_area = new QWidget(splitter); // NOLINT(cppcoreguidelines-owning-memory)
    board_area->setAccessibleName(QStringLiteral("Board area"));
    auto* board_layout = new QVBoxLayout(board_area); // NOLINT(cppcoreguidelines-owning-memory)
    board_layout->setContentsMargins(0, 0, 0, 0);
    board_layout->setSpacing(0);

    auto* board_row = new QWidget(board_area);           // NOLINT(cppcoreguidelines-owning-memory)
    auto* board_row_layout = new QHBoxLayout(board_row); // NOLINT(cppcoreguidelines-owning-memory)
    board_row_layout->setContentsMargins(0, 0, 0, 0);
    board_row_layout->setSpacing(kContentMargin / 2);

    evaluation_bar_ = new EvaluationBarWidget(board_row); // NOLINT(cppcoreguidelines-owning-memory)
    board_row_layout->addWidget(evaluation_bar_, 0, Qt::AlignVCenter);

    board_widget_ = new BoardWidget(board_row); // NOLINT(cppcoreguidelines-owning-memory)
    board_widget_->setObjectName(QStringLiteral("gameBoard"));
    board_widget_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(board_widget_, &BoardWidget::boardGeometryChanged, this, [this](QRect board_rect) {
        if (evaluation_bar_ != nullptr) {
            evaluation_bar_->set_reference_height(board_rect.height());
        }
    });
    connect(board_widget_, &BoardWidget::squareClicked, this,
            [this](int square_index) { handle_square_clicked(square_index); });
    connect(board_widget_, &BoardWidget::moveGestureCompleted, this,
            [this](int from_square_index, int to_square_index) {
                handle_move_gesture(from_square_index, to_square_index);
            });
    board_row_layout->addWidget(board_widget_, 1, Qt::AlignCenter);
    board_layout->addWidget(board_row, 1);

    game_sidebar_ = new GameSidebar(splitter); // NOLINT(cppcoreguidelines-owning-memory)
    game_sidebar_->set_navigation_actions(first_move_action_, previous_move_action_,
                                          next_move_action_, last_move_action_);
    connect(game_sidebar_, &GameSidebar::playModeChangeRequested, this,
            [this](GameSidebar::PlayMode mode) { set_play_mode(mode); });
    connect(game_sidebar_, &GameSidebar::newGameRequested, this, [this]() { new_game(); });
    connect(game_sidebar_, &GameSidebar::analysisToggleRequested, this,
            [this]() { toggle_analysis_pause(); });
    connect(game_sidebar_, &GameSidebar::stopSearchRequested, this,
            [this]() { stop_engine_search(); });

    splitter->addWidget(board_area);
    splitter->addWidget(game_sidebar_);
    splitter->setStretchFactor(0, 4);
    splitter->setStretchFactor(1, 1);
    layout->addWidget(splitter, 1);
    setCentralWidget(root);
}

void MainWindow::open_pgn() {
    const QString path = QFileDialog::getOpenFileName(
        this, QStringLiteral("Open PGN"), pgn_path_.isEmpty() ? QDir::homePath() : pgn_path_,
        QStringLiteral("PGN Files (*.pgn);;All Files (*)"));
    if (path.isEmpty()) {
        return;
    }

    QFile file{path};
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("Open PGN"),
                             QStringLiteral("Cannot open PGN: %1").arg(file.errorString()));
        return;
    }
    const QByteArray bytes = file.read(static_cast<qint64>(gui::pgn::max_pgn_bytes) + 1);
    if (bytes.size() > static_cast<qsizetype>(gui::pgn::max_pgn_bytes) || !file.atEnd()) {
        QMessageBox::warning(this, QStringLiteral("Open PGN"),
                             QStringLiteral("PGN exceeds the supported size limit."));
        return;
    }

    const auto parsed = gui::pgn::parse_game(
        std::string_view{bytes.constData(), static_cast<std::size_t>(bytes.size())});
    if (!parsed.has_value()) {
        QMessageBox::warning(this, QStringLiteral("Open PGN"),
                             QStringLiteral("Invalid PGN: %1").arg(pgn_error_text(parsed.error())));
        return;
    }
    if (!game_.load_pgn_game(*parsed)) {
        QMessageBox::warning(this, QStringLiteral("Open PGN"),
                             QStringLiteral("PGN move list could not be loaded safely."));
        return;
    }

    leave_engine_play_mode_for_pgn();
    pgn_path_ = path;
    const QString warning = pgn_warning_text(*parsed);
    sync_board_from_game(
        warning.isEmpty() ? QStringLiteral("Loaded PGN. %1").arg(game_status_text()) : warning);
}

void MainWindow::save_pgn() {
    if (pgn_path_.isEmpty()) {
        save_pgn_as();
        return;
    }
    (void)save_pgn_to_path(pgn_path_);
}

void MainWindow::save_pgn_as() {
    QString path = QFileDialog::getSaveFileName(this, QStringLiteral("Save PGN"),
                                                pgn_path_.isEmpty() ? QDir::homePath() : pgn_path_,
                                                QStringLiteral("PGN Files (*.pgn);;All Files (*)"));
    if (path.isEmpty()) {
        return;
    }
    QFileInfo info{path};
    if (info.suffix().isEmpty()) {
        path.append(QStringLiteral(".pgn"));
    }
    (void)save_pgn_to_path(path);
}

bool MainWindow::save_pgn_to_path(const QString& path) {
    const auto text = gui::pgn::write_game(game_.to_pgn_game());
    if (!text.has_value()) {
        QMessageBox::warning(
            this, QStringLiteral("Save PGN"),
            QStringLiteral("Cannot export PGN: %1").arg(pgn_error_text(text.error())));
        return false;
    }

    QSaveFile file{path};
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("Save PGN"),
                             QStringLiteral("Cannot write PGN: %1").arg(file.errorString()));
        return false;
    }
    const QByteArray bytes{text->data(), static_cast<qsizetype>(text->size())};
    if (file.write(bytes) != bytes.size()) {
        QMessageBox::warning(this, QStringLiteral("Save PGN"),
                             QStringLiteral("Cannot write PGN: %1").arg(file.errorString()));
        return false;
    }
    if (!file.commit()) {
        QMessageBox::warning(this, QStringLiteral("Save PGN"),
                             QStringLiteral("Cannot commit PGN: %1").arg(file.errorString()));
        return false;
    }
    pgn_path_ = path;
    sync_board_from_game(QStringLiteral("Saved PGN. %1").arg(game_status_text()));
    return true;
}

void MainWindow::navigate_first() {
    leave_engine_play_mode_for_pgn();
    if (game_.go_to_ply(0)) {
        sync_board_from_game(QStringLiteral("Showing start position. %1").arg(game_status_text()));
    }
}

void MainWindow::navigate_previous() {
    if (game_.current_ply() == 0U) {
        return;
    }
    leave_engine_play_mode_for_pgn();
    if (game_.go_to_ply(game_.current_ply() - 1U)) {
        sync_board_from_game(QStringLiteral("Showing move %1 of %2. %3")
                                 .arg(static_cast<qulonglong>(game_.current_ply()))
                                 .arg(static_cast<qulonglong>(game_.move_count()))
                                 .arg(game_status_text()));
    }
}

void MainWindow::navigate_next() {
    if (game_.current_ply() >= game_.move_count()) {
        return;
    }
    leave_engine_play_mode_for_pgn();
    if (game_.go_to_ply(game_.current_ply() + 1U)) {
        sync_board_from_game(QStringLiteral("Showing move %1 of %2. %3")
                                 .arg(static_cast<qulonglong>(game_.current_ply()))
                                 .arg(static_cast<qulonglong>(game_.move_count()))
                                 .arg(game_status_text()));
    }
}

void MainWindow::navigate_last() {
    leave_engine_play_mode_for_pgn();
    if (game_.go_to_ply(game_.move_count())) {
        sync_board_from_game(QStringLiteral("Showing final position. %1").arg(game_status_text()));
    }
}

void MainWindow::start_engine() {
    if (engine_controller_ == nullptr) {
        return;
    }
    apply_settings_to_engine();
    if (!engine_controller_->start_engine()) {
        sync_board_from_game(
            QStringLiteral("Engine could not start. %1").arg(engine_controller_->engine_path()));
        return;
    }
    sync_board_from_game(QStringLiteral("Starting engine. %1").arg(engine_status_text()));
    update_engine_actions();
}

void MainWindow::stop_engine() {
    awaiting_engine_move_ = false;
    pending_engine_turn_ = false;
    pending_analysis_ = false;
    restart_engine_when_stopped_ = false;
    active_search_purpose_ = SearchPurpose::None;
    active_search_side_to_move_.reset();
    if (engine_controller_ != nullptr) {
        engine_controller_->quit_engine();
    }
    sync_board_from_game(QStringLiteral("Stopping engine."));
    update_engine_actions();
}

void MainWindow::restart_engine() {
    if (engine_controller_ == nullptr) {
        return;
    }
    const bool resume_analysis_after_restart =
        active_search_purpose_ == SearchPurpose::Analysis || pending_analysis_;
    awaiting_engine_move_ = false;
    pending_engine_turn_ = engine_to_move();
    pending_analysis_ = resume_analysis_after_restart && !analysis_paused_ &&
                        !game_.status().terminal && !engine_to_move();
    active_search_purpose_ = SearchPurpose::None;
    active_search_side_to_move_.reset();
    restart_engine_when_stopped_ = true;
    if (engine_controller_->is_running()) {
        engine_controller_->quit_engine();
    } else {
        restart_engine_when_stopped_ = false;
        start_engine();
    }
    update_engine_actions();
}

void MainWindow::stop_engine_search() {
    awaiting_engine_move_ = false;
    pending_engine_turn_ = false;
    pending_analysis_ = false;
    active_search_purpose_ = SearchPurpose::None;
    active_search_side_to_move_.reset();
    if (engine_controller_ != nullptr) {
        engine_controller_->stop_search();
    }
    sync_board_from_game(QStringLiteral("Engine search stopped. %1").arg(game_status_text()));
    update_engine_actions();
}

void MainWindow::toggle_analysis_pause() {
    const bool analysis_active =
        active_search_purpose_ == SearchPurpose::Analysis || pending_analysis_;
    if (analysis_paused_ || !analysis_active) {
        resume_analysis();
    } else {
        pause_analysis();
    }
}

void MainWindow::pause_analysis() {
    analysis_paused_ = true;
    pending_analysis_ = false;
    if (active_search_purpose_ == SearchPurpose::Analysis && engine_controller_ != nullptr) {
        active_search_purpose_ = SearchPurpose::None;
        active_search_side_to_move_.reset();
        engine_controller_->stop_search();
    }
    sync_board_from_game(QStringLiteral("Analysis paused. %1").arg(game_status_text()));
}

void MainWindow::resume_analysis() {
    analysis_paused_ = false;
    if (game_.status().terminal) {
        sync_board_from_game(QStringLiteral("No analyzable position. %1").arg(game_status_text()));
        return;
    }
    request_engine_move_if_needed();
    refresh_analysis_after_position_change();
}

void MainWindow::open_preferences() {
    PreferencesDialog dialog{settings_, this};
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    const QString previous_engine_path = settings_.engine_path;
    settings_ = dialog.settings();
    settings_.save();
    apply_settings_to_engine();

    const bool path_changed = previous_engine_path != settings_.engine_path;
    if (path_changed && engine_controller_ != nullptr && engine_controller_->is_running()) {
        sync_board_from_game(
            QStringLiteral("Preferences saved. Restart engine to use the new executable."));
    } else {
        sync_board_from_game(QStringLiteral("Preferences saved. %1").arg(game_status_text()));
    }
}

void MainWindow::set_play_mode(GameSidebar::PlayMode mode, bool require_confirmation) {
    if (mode == GameSidebar::PlayMode::HumanVsEngine && require_confirmation &&
        play_mode_ != GameSidebar::PlayMode::HumanVsEngine) {
        const auto choice = QMessageBox::warning(
            this, QStringLiteral("Start Engine"),
            QStringLiteral("Human vs Engine will start the local engine process. Searches can use "
                           "significant CPU and memory while the engine is thinking."),
            QMessageBox::Ok | QMessageBox::Cancel, QMessageBox::Cancel);
        if (choice != QMessageBox::Ok) {
            if (game_sidebar_ != nullptr) {
                game_sidebar_->set_play_mode(play_mode_);
            }
            if (play_engine_action_ != nullptr) {
                play_engine_action_->setChecked(play_mode_ == GameSidebar::PlayMode::HumanVsEngine);
            }
            return;
        }
    }

    const bool was_engine_mode = play_mode_ == GameSidebar::PlayMode::HumanVsEngine;
    const bool enabled = mode == GameSidebar::PlayMode::HumanVsEngine;
    play_mode_ = mode;
    awaiting_engine_move_ = false;
    pending_engine_turn_ = false;
    pending_analysis_ = false;
    if (!enabled && was_engine_mode && active_search_purpose_ == SearchPurpose::EngineMove &&
        engine_controller_ != nullptr &&
        engine_controller_->state() == EngineController::State::Searching) {
        active_search_purpose_ = SearchPurpose::None;
        active_search_side_to_move_.reset();
        engine_controller_->stop_search();
    }
    if (enabled && engine_controller_ != nullptr &&
        (engine_controller_->state() == EngineController::State::Stopped ||
         engine_controller_->state() == EngineController::State::Failed)) {
        pending_engine_turn_ = engine_to_move();
        start_engine();
    }
    sync_board_from_game(enabled ? QStringLiteral("Play vs engine enabled. Human plays white.")
                                 : QStringLiteral("Human vs human mode."));
    request_engine_move_if_needed();
    if (!enabled && active_search_purpose_ != SearchPurpose::Analysis) {
        refresh_analysis_after_position_change();
    }
    update_engine_actions();
}

void MainWindow::leave_engine_play_mode_for_pgn() {
    awaiting_engine_move_ = false;
    pending_engine_turn_ = false;
    pending_analysis_ = false;
    active_search_purpose_ = SearchPurpose::None;
    active_search_side_to_move_.reset();
    if (engine_controller_ != nullptr &&
        engine_controller_->state() == EngineController::State::Searching) {
        engine_controller_->stop_search();
    }
    play_mode_ = GameSidebar::PlayMode::HumanVsHuman;
    if (play_engine_action_ != nullptr && play_engine_action_->isChecked()) {
        play_engine_action_->setChecked(false);
    }
    if (game_sidebar_ != nullptr) {
        game_sidebar_->set_play_mode(play_mode_);
    }
    update_engine_actions();
}

void MainWindow::new_game() {
    game_.new_game();
    pgn_path_.clear();
    awaiting_engine_move_ = false;
    pending_engine_turn_ = false;
    pending_analysis_ = false;
    active_search_purpose_ = SearchPurpose::None;
    active_search_side_to_move_.reset();
    if (engine_controller_ != nullptr && engine_controller_->is_running()) {
        engine_controller_->new_game();
    }
    if (game_sidebar_ != nullptr) {
        game_sidebar_->clear_for_new_game();
    }
    if (evaluation_bar_ != nullptr) {
        evaluation_bar_->reset_to_neutral();
    }
    sync_board_from_game(QStringLiteral("New game. %1").arg(game_status_text()));
    request_engine_move_if_needed();
}

void MainWindow::handle_square_clicked(int square_index) {
    const auto square = eng::Square::from_index(square_index);
    if (!square.is_valid()) {
        return;
    }
    if (!human_can_move()) {
        if (!game_.is_at_latest_ply()) {
            sync_board_from_game(QStringLiteral("Navigate to the final move to continue playing."));
        } else {
            sync_board_from_game(awaiting_engine_move_ ? QStringLiteral("Engine is thinking.")
                                                       : QStringLiteral("Engine to move."));
        }
        return;
    }
    if (game_.status().terminal) {
        sync_board_from_game(QStringLiteral("Game over. Start a new game to continue."));
        return;
    }

    const auto selected = game_.selected_square();
    if (selected.has_value() && game_.destination_is_legal(square)) {
        handle_move_attempt(*selected, square);
        return;
    }

    if (game_.select_square(square)) {
        sync_board_from_game();
        return;
    }

    game_.clear_selection();
    sync_board_from_game(
        QStringLiteral("Select a legal %1 piece. %2")
            .arg(color_name(game_.position().side_to_move()).toLower(), game_status_text()));
}

void MainWindow::handle_move_gesture(int from_square_index, int to_square_index) {
    const auto from_square = eng::Square::from_index(from_square_index);
    const auto to_square = eng::Square::from_index(to_square_index);
    if (!from_square.is_valid() || !to_square.is_valid()) {
        return;
    }
    if (!human_can_move()) {
        if (!game_.is_at_latest_ply()) {
            sync_board_from_game(QStringLiteral("Navigate to the final move to continue playing."));
        } else {
            sync_board_from_game(awaiting_engine_move_ ? QStringLiteral("Engine is thinking.")
                                                       : QStringLiteral("Engine to move."));
        }
        return;
    }
    handle_move_attempt(from_square, to_square);
}

void MainWindow::handle_move_attempt(eng::Square from_square, eng::Square to_square) {
    auto attempt = game_.try_move(from_square, to_square);
    if (attempt.kind == GuiGameState::MoveAttemptKind::PromotionRequired) {
        const auto promotion = choose_promotion_piece(attempt);
        if (!promotion.has_value()) {
            game_.clear_selection();
            sync_board_from_game(QStringLiteral("Promotion cancelled. %1").arg(game_status_text()));
            return;
        }
        attempt = game_.try_move(from_square, to_square, promotion);
    }

    switch (attempt.kind) {
    case GuiGameState::MoveAttemptKind::Applied:
        sync_board_from_game();
        request_engine_move_if_needed();
        refresh_analysis_after_position_change();
        return;
    case GuiGameState::MoveAttemptKind::GameOver:
        sync_board_from_game(QStringLiteral("Game over. Start a new game to continue."));
        return;
    case GuiGameState::MoveAttemptKind::PromotionRequired:
    case GuiGameState::MoveAttemptKind::Invalid:
        game_.clear_selection();
        sync_board_from_game(QStringLiteral("Illegal move. %1").arg(game_status_text()));
        return;
    }
}

void MainWindow::handle_engine_state_changed() {
    if (engine_controller_ == nullptr) {
        return;
    }
    update_engine_actions();

    const auto state = engine_controller_->state();
    if (game_sidebar_ != nullptr) {
        game_sidebar_->set_engine_state(state);
    }
    if (state == EngineController::State::Stopped && restart_engine_when_stopped_) {
        restart_engine_when_stopped_ = false;
        QTimer::singleShot(0, this, [this]() { start_engine(); });
        return;
    }
    if (state == EngineController::State::Ready) {
        if (pending_engine_turn_ || !pending_analysis_) {
            request_engine_move_if_needed();
        } else {
            pending_analysis_ = false;
            start_analysis_for_current_position();
        }
    } else if (state == EngineController::State::Failed) {
        awaiting_engine_move_ = false;
        pending_engine_turn_ = false;
        pending_analysis_ = false;
        restart_engine_when_stopped_ = false;
        active_search_purpose_ = SearchPurpose::None;
        active_search_side_to_move_.reset();
    }
    sync_board_from_game();
}

void MainWindow::handle_engine_search_info(const EngineController::SearchInfo& info) {
    const eng::Color side_to_move =
        active_search_side_to_move_.value_or(game_.position().side_to_move());
    if (evaluation_bar_ != nullptr && info.multipv == 1) {
        evaluation_bar_->set_evaluation(
            gui::evaluation_display_from_search_info(info, side_to_move));
    }
    if (game_sidebar_ != nullptr) {
        game_sidebar_->record_search_info(info, side_to_move);
    }
}

void MainWindow::handle_engine_event(const QString& event_type, const QString& detail) {
    if (game_sidebar_ != nullptr) {
        game_sidebar_->append_engine_event(event_type, detail);
    }
}

void MainWindow::handle_engine_bestmove(const QString& notation) {
    if (active_search_purpose_ == SearchPurpose::Analysis) {
        active_search_purpose_ = SearchPurpose::None;
        active_search_side_to_move_.reset();
        return;
    }
    if (active_search_purpose_ != SearchPurpose::EngineMove || !awaiting_engine_move_) {
        return;
    }
    awaiting_engine_move_ = false;
    pending_engine_turn_ = false;
    active_search_purpose_ = SearchPurpose::None;
    active_search_side_to_move_.reset();

    if (notation == QStringLiteral("0000")) {
        sync_board_from_game(
            QStringLiteral("Engine returned no legal move. %1").arg(game_status_text()));
        return;
    }

    const auto parsed = legal_move_from_uci(game_.position(), notation);
    if (!parsed.has_value()) {
        sync_board_from_game(QStringLiteral("Engine returned illegal bestmove %1. %2")
                                 .arg(notation, game_status_text()));
        return;
    }

    const std::optional<eng::PieceType> promotion =
        parsed->is_promotion() ? std::optional<eng::PieceType>{parsed->promotion_piece()}
                               : std::nullopt;
    const auto attempt = game_.try_move(parsed->from(), parsed->to(), promotion);
    if (attempt.kind != GuiGameState::MoveAttemptKind::Applied) {
        sync_board_from_game(QStringLiteral("Engine move %1 could not be applied. %2")
                                 .arg(notation, game_status_text()));
        return;
    }
    sync_board_from_game(QStringLiteral("Engine played %1. %2").arg(notation, game_status_text()));
    refresh_analysis_after_position_change();
}

void MainWindow::request_engine_move_if_needed() {
    if (play_mode_ != GameSidebar::PlayMode::HumanVsEngine || !game_.is_at_latest_ply() ||
        game_.status().terminal || !engine_to_move()) {
        pending_engine_turn_ = false;
        return;
    }
    if (awaiting_engine_move_) {
        return;
    }
    if (engine_controller_ == nullptr) {
        return;
    }

    const auto state = engine_controller_->state();
    if (state == EngineController::State::Ready) {
        start_engine_search_for_current_position();
        return;
    }
    if (state == EngineController::State::Stopped || state == EngineController::State::Failed) {
        pending_engine_turn_ = true;
        start_engine();
        return;
    }
    if (state == EngineController::State::Starting) {
        pending_engine_turn_ = true;
        return;
    }
    if (state == EngineController::State::Searching || state == EngineController::State::Stopping) {
        pending_engine_turn_ = true;
        if (active_search_purpose_ == SearchPurpose::Analysis &&
            state == EngineController::State::Searching) {
            active_search_purpose_ = SearchPurpose::None;
            active_search_side_to_move_.reset();
            engine_controller_->stop_search();
        }
    }
}

void MainWindow::start_engine_search_for_current_position() {
    if (engine_controller_ == nullptr || game_.status().terminal || !engine_to_move()) {
        return;
    }

    const QString fen = QString::fromStdString(game_.position().to_fen());
    if (!engine_controller_->set_position_fen(fen)) {
        sync_board_from_game(QStringLiteral("Engine is not ready for a position update."));
        return;
    }
    if (game_sidebar_ != nullptr) {
        game_sidebar_->begin_search(
            QStringLiteral("engine move %1 ms").arg(settings_.engine_move_time_ms));
    }
    if (evaluation_bar_ != nullptr) {
        evaluation_bar_->reset_to_neutral();
    }
    if (!engine_controller_->go_movetime(settings_.engine_move_time_ms)) {
        sync_board_from_game(QStringLiteral("Engine search could not start."));
        return;
    }
    active_search_purpose_ = SearchPurpose::EngineMove;
    active_search_side_to_move_ = game_.position().side_to_move();
    awaiting_engine_move_ = true;
    pending_engine_turn_ = false;
    sync_board_from_game(
        QStringLiteral("Engine thinking for %1 ms.").arg(settings_.engine_move_time_ms));
    update_engine_actions();
}

void MainWindow::start_analysis_for_current_position() {
    if (engine_controller_ == nullptr || analysis_paused_ || game_.status().terminal) {
        return;
    }

    const QString fen = QString::fromStdString(game_.position().to_fen());
    if (!engine_controller_->set_position_fen(fen)) {
        sync_board_from_game(QStringLiteral("Engine is not ready for analysis."));
        return;
    }
    if (game_sidebar_ != nullptr) {
        game_sidebar_->begin_search(QStringLiteral("continuous analysis"));
    }
    if (evaluation_bar_ != nullptr) {
        evaluation_bar_->reset_to_neutral();
    }
    if (!engine_controller_->go_infinite()) {
        sync_board_from_game(QStringLiteral("Engine analysis could not start."));
        return;
    }
    active_search_purpose_ = SearchPurpose::Analysis;
    active_search_side_to_move_ = game_.position().side_to_move();
    pending_analysis_ = false;
    sync_board_from_game(QStringLiteral("Analyzing current position."));
    update_engine_actions();
}

void MainWindow::refresh_analysis_after_position_change() {
    if (analysis_paused_ || game_.status().terminal) {
        pending_analysis_ = false;
        if (active_search_purpose_ == SearchPurpose::Analysis && engine_controller_ != nullptr) {
            active_search_purpose_ = SearchPurpose::None;
            active_search_side_to_move_.reset();
            engine_controller_->stop_search();
        }
        return;
    }
    if (play_mode_ == GameSidebar::PlayMode::HumanVsEngine && engine_to_move()) {
        return;
    }
    request_analysis_for_current_position();
}

void MainWindow::request_analysis_for_current_position() {
    if (engine_controller_ == nullptr || analysis_paused_ || game_.status().terminal) {
        pending_analysis_ = false;
        return;
    }

    const auto state = engine_controller_->state();
    if (state == EngineController::State::Ready) {
        start_analysis_for_current_position();
        return;
    }
    if (state == EngineController::State::Stopped || state == EngineController::State::Failed) {
        pending_analysis_ = true;
        start_engine();
        return;
    }
    if (state == EngineController::State::Starting) {
        pending_analysis_ = true;
        sync_board_from_game(QStringLiteral("Analysis queued until engine is ready."));
        return;
    }
    if (state == EngineController::State::Searching || state == EngineController::State::Stopping) {
        pending_analysis_ = true;
        if (active_search_purpose_ == SearchPurpose::Analysis &&
            state == EngineController::State::Searching) {
            engine_controller_->stop_search();
        }
    }
}

bool MainWindow::human_can_move() const {
    if (game_.status().terminal) {
        return false;
    }
    if (play_mode_ != GameSidebar::PlayMode::HumanVsEngine) {
        return game_.is_at_latest_ply();
    }
    return game_.is_at_latest_ply() && !awaiting_engine_move_ && !engine_to_move();
}

bool MainWindow::engine_to_move() const noexcept {
    return play_mode_ == GameSidebar::PlayMode::HumanVsEngine &&
           game_.position().side_to_move() == kEngineColor;
}

std::optional<eng::PieceType>
MainWindow::choose_promotion_piece(const GuiGameState::MoveAttempt& attempt) {
    QMessageBox box{this};
    box.setWindowTitle(QStringLiteral("Promote Pawn"));
    box.setText(QStringLiteral("Choose promotion piece"));
    box.setIcon(QMessageBox::Question);
    box.setWindowModality(Qt::WindowModal);

    std::array<QPushButton*, 4> buttons{};
    for (std::size_t index = 0; index < attempt.promotion_count && index < buttons.size();
         ++index) {
        buttons[index] =
            box.addButton(promotion_name(attempt.promotion_pieces[index]), QMessageBox::AcceptRole);
        if (index == 0U) {
            box.setDefaultButton(buttons[index]);
        }
    }
    box.addButton(QMessageBox::Cancel);
    box.exec();

    const auto* clicked = box.clickedButton();
    for (std::size_t index = 0; index < attempt.promotion_count && index < buttons.size();
         ++index) {
        if (clicked == buttons[index]) {
            return attempt.promotion_pieces[index];
        }
    }
    return std::nullopt;
}

void MainWindow::sync_board_from_game(const QString& status_override) {
    if (board_widget_ == nullptr) {
        return;
    }

    board_widget_->set_position(game_.position());
    board_widget_->set_active_side(game_.position().side_to_move());
    board_widget_->set_last_move(game_.last_move());
    board_widget_->set_selected_square(game_.selected_square());
    board_widget_->set_legal_destinations(game_.legal_destinations());
    board_widget_->set_check_square(game_.check_square());
    const QString game_text = status_override.isEmpty() ? game_status_text() : status_override;
    statusBar()->showMessage(QStringLiteral("%1 | %2").arg(game_text, engine_status_text()));
    if (game_sidebar_ != nullptr) {
        game_sidebar_->set_game_snapshot(GameSidebar::GameSnapshot{
            .play_mode = play_mode_,
            .game_status = game_text,
            .engine_status = engine_status_text(),
            .awaiting_engine_move = awaiting_engine_move_,
            .engine_to_move = engine_to_move(),
            .terminal = game_.status().terminal,
        });
    }
    update_engine_actions();
    update_pgn_actions();
}

QString MainWindow::game_status_text() const {
    const auto& status = game_.status();
    switch (status.kind) {
    case GuiGameState::StatusKind::Playing:
        return QStringLiteral("%1 to move").arg(color_name(status.side_to_move));
    case GuiGameState::StatusKind::Check:
        return QStringLiteral("%1 to move - check").arg(color_name(status.side_to_move));
    case GuiGameState::StatusKind::Checkmate:
        return QStringLiteral("Checkmate. %1 wins.").arg(color_name(status.winner));
    case GuiGameState::StatusKind::Stalemate:
        return QStringLiteral("Draw by stalemate.");
    case GuiGameState::StatusKind::DrawFiftyMove:
        return QStringLiteral("Draw by 50-move rule.");
    case GuiGameState::StatusKind::DrawRepetition:
        return QStringLiteral("Draw by threefold repetition.");
    case GuiGameState::StatusKind::DrawInsufficientMaterial:
        return QStringLiteral("Draw by insufficient material.");
    }
    return QStringLiteral("Ready");
}

QString MainWindow::engine_status_text() const {
    if (engine_controller_ == nullptr) {
        return QStringLiteral("Engine: unavailable");
    }
    const auto state = engine_controller_->state();
    QString status = QStringLiteral("Engine: %1").arg(EngineController::state_name(state));
    if (play_mode_ == GameSidebar::PlayMode::HumanVsEngine) {
        status.append(QStringLiteral(" | Mode: vs engine"));
    }
    status.append(analysis_paused_ ? QStringLiteral(" | Analysis: paused")
                                   : QStringLiteral(" | Analysis: auto"));
    return status;
}

QString MainWindow::default_engine_path() {
    const QString executable_name = QStringLiteral("localchessengine");
    const QDir app_dir{QCoreApplication::applicationDirPath()};
    QStringList candidates;
    candidates.push_back(app_dir.absoluteFilePath(executable_name));

    QDir build_dir = app_dir;
    if (build_dir.dirName() == QStringLiteral("gui") && build_dir.cdUp()) {
        candidates.push_back(
            build_dir.absoluteFilePath(QStringLiteral("engine/%1").arg(executable_name)));
    }

    QDir bundle_build_dir = app_dir;
    for (int index = 0; index < kBundleBuildDirUpLevels; ++index) {
        if (!bundle_build_dir.cdUp()) {
            break;
        }
    }
    candidates.push_back(
        bundle_build_dir.absoluteFilePath(QStringLiteral("engine/%1").arg(executable_name)));

    candidates.push_back(QDir::current().absoluteFilePath(
        QStringLiteral("build/debug/engine/%1").arg(executable_name)));
    candidates.push_back(QDir::current().absoluteFilePath(
        QStringLiteral("build/release/engine/%1").arg(executable_name)));

    for (const auto& candidate : candidates) {
        const QFileInfo info{candidate};
        if (info.exists() && info.isFile() && info.isExecutable()) {
            return info.absoluteFilePath();
        }
    }
    return candidates.isEmpty() ? QString{} : candidates.front();
}

EngineController::EngineOptions
MainWindow::engine_options_from_settings(const GuiSettings& settings) {
    return EngineController::EngineOptions{
        .hash_mb = settings.hash_mb,
        .threads = settings.threads,
        .multi_pv = settings.multi_pv,
        .move_overhead_ms = settings.move_overhead_ms,
        .use_nnue = settings.use_nnue,
        .eval_file = settings.eval_file,
        .nnue_backend = settings.nnue_backend_uci_name(),
    };
}

void MainWindow::apply_settings_to_engine() {
    if (engine_controller_ == nullptr) {
        return;
    }
    engine_controller_->set_engine_path(settings_.engine_path);
    engine_controller_->set_engine_options(engine_options_from_settings(settings_));
}

void MainWindow::update_engine_actions() {
    if (engine_controller_ == nullptr) {
        return;
    }
    const auto state = engine_controller_->state();
    const bool stopped =
        state == EngineController::State::Stopped || state == EngineController::State::Failed;
    const bool running =
        state == EngineController::State::Starting || state == EngineController::State::Ready ||
        state == EngineController::State::Searching || state == EngineController::State::Stopping;

    if (start_engine_action_ != nullptr) {
        start_engine_action_->setEnabled(stopped);
    }
    const bool can_toggle_analysis = !game_.status().terminal;
    const bool analysis_active =
        active_search_purpose_ == SearchPurpose::Analysis || pending_analysis_;
    if (analysis_toggle_action_ != nullptr) {
        analysis_toggle_action_->setEnabled(can_toggle_analysis);
        if (analysis_paused_) {
            analysis_toggle_action_->setText(QStringLiteral("&Resume Analysis"));
        } else if (analysis_active) {
            analysis_toggle_action_->setText(QStringLiteral("&Pause Analysis"));
        } else {
            analysis_toggle_action_->setText(QStringLiteral("&Analyze Current Position"));
        }
    }
    if (stop_search_action_ != nullptr) {
        stop_search_action_->setEnabled(state == EngineController::State::Searching ||
                                        state == EngineController::State::Stopping);
    }
    if (stop_engine_action_ != nullptr) {
        stop_engine_action_->setEnabled(running);
    }
    if (restart_engine_action_ != nullptr) {
        restart_engine_action_->setEnabled(true);
    }
    if (play_engine_action_ != nullptr &&
        play_engine_action_->isChecked() != (play_mode_ == GameSidebar::PlayMode::HumanVsEngine)) {
        play_engine_action_->setChecked(play_mode_ == GameSidebar::PlayMode::HumanVsEngine);
    }
    if (game_sidebar_ != nullptr) {
        game_sidebar_->set_engine_state(state);
        game_sidebar_->set_controls(GameSidebar::Controls{
            .can_toggle_analysis = can_toggle_analysis,
            .analysis_paused = analysis_paused_,
            .analysis_active = analysis_active,
            .can_stop_search = state == EngineController::State::Searching ||
                               state == EngineController::State::Stopping,
        });
    }
}

void MainWindow::update_pgn_actions() {
    if (save_pgn_action_ != nullptr) {
        save_pgn_action_->setEnabled(true);
    }
    if (save_pgn_as_action_ != nullptr) {
        save_pgn_as_action_->setEnabled(true);
    }
    if (first_move_action_ != nullptr) {
        first_move_action_->setEnabled(game_.current_ply() > 0U);
    }
    if (previous_move_action_ != nullptr) {
        previous_move_action_->setEnabled(game_.current_ply() > 0U);
    }
    if (next_move_action_ != nullptr) {
        next_move_action_->setEnabled(game_.current_ply() < game_.move_count());
    }
    if (last_move_action_ != nullptr) {
        last_move_action_->setEnabled(game_.current_ply() < game_.move_count());
    }
}

void MainWindow::install_status_bar_branding() {
    constexpr int kBrandingIconPx = 18;
    auto* branding = new QLabel(statusBar()); // NOLINT(cppcoreguidelines-owning-memory)
    branding->setPixmap(render_app_icon_pixmap(kBrandingIconPx, devicePixelRatioF()));
    branding->setFixedSize(kBrandingIconPx, kBrandingIconPx);
    branding->setToolTip(QStringLiteral("LocalChessEngine"));
    branding->setContentsMargins(0, 0, 0, 0);
    statusBar()->addPermanentWidget(branding);
}

void MainWindow::show_about_dialog() {
    constexpr int kAboutIconPx = 128;
    QMessageBox dialog(this);
    dialog.setWindowTitle(QStringLiteral("About LocalChessEngine"));
    dialog.setTextFormat(Qt::PlainText);
    dialog.setIconPixmap(render_app_icon_pixmap(kAboutIconPx, devicePixelRatioF()));
    dialog.setText(
        QStringLiteral("LocalChessEngine %1").arg(QCoreApplication::applicationVersion()));
    dialog.setInformativeText(QStringLiteral(
        "Local-first chess engine and Qt Widgets desktop shell for macOS on Apple Silicon.\n\n"
        "The UCI engine remains available as a separate command-line executable."));
    dialog.setStandardButtons(QMessageBox::Ok);
    dialog.exec();
}

void MainWindow::show_license_dialog() {
    QMessageBox::information(
        this, QStringLiteral("Third-Party Notices"),
        QStringLiteral(
            "Qt 6 Core, Gui, Widgets, Svg and SvgWidgets are dynamically linked under the Qt "
            "LGPLv3/commercial licensing model.\n\n"
            "Project code and self-authored SVG chess pieces are licensed under MIT.\n\n"
            "Release bundles include full notices, license texts, Qt relink instructions, and "
            "corresponding source materials under Contents/Resources/Legal."));
}
